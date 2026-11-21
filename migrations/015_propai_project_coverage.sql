-- Two fixes to project coverage (run the whole file; it is one transaction).
-- 1. Sales and rents whose own project link is empty are linked through their DLD project number (off-plan launches such as
--    Lunaya by Zaya sell before Prop-AI links them), so projects get their real prices and volumes.
-- 2. Projects that only appear in sales/rent records (towers and buildings missing from Prop-AI's project list, e.g. West Heights 1)
--    are added as projects, built from those records: name, community, master project, use and prices, but no photos or description.
-- Refresh after a big reload: REFRESH MATERIALIZED VIEW public.propai_project_prices; then public.propai_derived_projects; then public.propai_area_stats;
BEGIN;
DROP MATERIALIZED VIEW IF EXISTS public.propai_area_stats;
DROP VIEW IF EXISTS public.propai_projects;
DROP VIEW IF EXISTS public.propai_projects_dld;
DROP MATERIALIZED VIEW IF EXISTS public.propai_derived_projects;
DROP MATERIALIZED VIEW IF EXISTS public.propai_project_prices;

CREATE MATERIALIZED VIEW public.propai_project_prices AS
WITH pn AS (
  SELECT DISTINCT ON (payload ->> 'project_number') payload ->> 'project_number' AS num, payload ->> 'prop-ai_location_key_project' AS lk
  FROM public.propai_raw
  WHERE dataset = 'projects' AND coalesce(payload ->> 'project_number', '') <> '' AND payload ->> 'prop-ai_location_key_project' IS NOT NULL
  ORDER BY payload ->> 'project_number', (coalesce(payload ->> 'project_name_en', '') ~* '\mplot\M')),
sl AS (
  SELECT coalesce(s.location_key_project, pn.lk) AS k, s.property_type, s.sale_price, s.price_per_sqft, s.instance_date, s.outlier_flag
  FROM public.propai_sales s LEFT JOIN pn ON pn.num = s.project_number WHERE s.deleted_at IS NULL),
rl AS (
  SELECT coalesce(r.location_key_project, pn.lk) AS k, r.property_type, r.contract_start_date
  FROM public.propai_rents r LEFT JOIN pn ON pn.num = r.project_number WHERE r.deleted_at IS NULL),
t AS (
  SELECT k, property_type AS type FROM sl WHERE k IS NOT NULL
  UNION ALL
  SELECT k, property_type FROM rl WHERE k IS NOT NULL
), n AS (SELECT k, type, count(*) c FROM t WHERE type IS NOT NULL AND type <> 'Other' GROUP BY 1, 2),
seg AS (
  SELECT k, (array_agg(type ORDER BY c DESC))[1] AS main_type, (array_agg(type ORDER BY c DESC))[1:3] AS types,
         (array_agg(CASE WHEN type IN ('Apartment','Villa','Hotel Apartment','Stacked Townhouse','Stacked Townhouses') THEN 'residential'
                         WHEN type = 'Office' THEN 'office' WHEN type IN ('Shop','Showroom','Warehouse','Building') THEN 'commercial'
                         WHEN type = 'Land' THEN 'land' END ORDER BY c DESC))[1] AS segment
  FROM n GROUP BY k),
r AS (SELECT k, count(*)::int AS rent_contracts FROM rl WHERE k IS NOT NULL AND contract_start_date <= current_date GROUP BY 1),
p AS (
  SELECT k, count(*)::int AS sales_count, avg(sale_price)::float8 AS avg_price, avg(price_per_sqft)::float8 AS avg_price_per_sqft,
         count(*) FILTER (WHERE instance_date >= current_date - interval '12 months')::int AS sales_12m,
         count(*) FILTER (WHERE instance_date < current_date - interval '12 months')::int AS sales_prev12m,
         (percentile_cont(0.5) WITHIN GROUP (ORDER BY sale_price) FILTER (WHERE instance_date >= current_date - interval '12 months'))::float8 AS median_price_12m,
         (percentile_cont(0.5) WITHIN GROUP (ORDER BY price_per_sqft) FILTER (WHERE instance_date >= current_date - interval '12 months'))::float8 AS median_ppsf_12m
  FROM sl
  WHERE k IS NOT NULL AND instance_date >= current_date - interval '24 months' AND instance_date <= current_date AND sale_price > 0
        AND coalesce(outlier_flag,'') <> 'True'
  GROUP BY 1)
SELECT coalesce(seg.k, p.k) AS location_key, p.sales_count, p.avg_price, p.avg_price_per_sqft, p.sales_12m, p.sales_prev12m, p.median_price_12m, p.median_ppsf_12m, seg.main_type, seg.types, seg.segment, coalesce(r.rent_contracts, 0) AS rent_contracts
FROM seg FULL JOIN p ON p.k = seg.k LEFT JOIN r ON r.k = coalesce(seg.k, p.k);
CREATE UNIQUE INDEX propai_project_prices_key ON public.propai_project_prices(location_key);

-- projects known only from sales/rent records
CREATE MATERIALIZED VIEW public.propai_derived_projects AS
WITH known AS (SELECT DISTINCT payload ->> 'prop-ai_location_key_project' AS k FROM public.propai_raw WHERE dataset = 'projects'),
t AS (
  SELECT location_key_project AS k, project_name_prop_ai AS name, developer_group, development_status FROM public.propai_sales WHERE location_key_project IS NOT NULL AND deleted_at IS NULL
  UNION ALL
  SELECT location_key_project, project_name_prop_ai, NULL, NULL FROM public.propai_rents WHERE location_key_project IS NOT NULL AND deleted_at IS NULL)
SELECT t.k, initcap(coalesce(max(t.name), split_part(t.k, '_', 3))) AS name, initcap(split_part(t.k, '_', 1)) AS area, initcap(split_part(t.k, '_', 2)) AS master_project,
       max(t.developer_group) AS developer_group, bool_or(t.development_status = 'Off-Plan') AS offplan
FROM t WHERE t.k NOT IN (SELECT k FROM known WHERE k IS NOT NULL) GROUP BY t.k;
CREATE UNIQUE INDEX propai_derived_projects_key ON public.propai_derived_projects(k);

-- One row per Prop-AI project, flattened from the raw /data/projects/basic-details payload (propai_raw, dataset='projects').
-- image_urls are presigned links that expire after 1 hour, so only the count is exposed here; images need copying to our own storage.
CREATE VIEW public.propai_projects_dld AS
SELECT r.row_key AS key,
       r.payload->>'prop-ai_location_key_project'      AS location_key,
       r.payload->>'project_id'                        AS project_id,
       r.payload->>'project_number'                    AS project_number,
       coalesce(r.payload->>'project_name_label', r.payload->>'project_name_en') AS name,
       coalesce(r.payload->>'area_name_label', r.payload->>'area_name_en')       AS area,
       coalesce(r.payload->>'master_project_label', r.payload->>'master_project_en') AS master_project,
       coalesce(r.payload->>'developer_name_en', r.payload->>'developer_name', st.payload->>'developer_name')   AS developer,
       r.payload->>'developer_group'                   AS developer_group,
       initcap(pp.segment)                           AS usage,
       CASE r.payload->>'development_status' WHEN 'FINISHED' THEN 'Completed' WHEN 'ACTIVE' THEN 'Under construction' WHEN 'PENDING' THEN 'Pending'
            WHEN 'NOT_STARTED' THEN 'Not started' WHEN 'CANCELLED' THEN 'Cancelled' WHEN 'FRIEZED' THEN 'Frozen' ELSE NULL END AS status_group,
       r.payload->>'development_status'                AS development_status,
       r.payload->>'offplan_status'                    AS offplan_status,
       r.payload->>'delivery_status'                   AS delivery_status,
       r.payload->>'data_status'                       AS data_status,
       coalesce(nullif(r.payload->>'percent_completed','')::numeric, nullif(st.payload->>'percent_completed','')::numeric) AS percent_completed,
       coalesce(nullif(r.payload->>'project_start_date','')::timestamp::date, nullif(st.payload->>'project_start_date','')::timestamp::date) AS start_date,
       coalesce(nullif(r.payload->>'project_end_date','')::timestamp::date, nullif(st.payload->>'project_end_date','')::timestamp::date) AS end_date,
       nullif(r.payload->>'project_completion_date','')::timestamp::date AS completion_date,
       r.payload->>'offplan_completion_year'           AS offplan_completion_year,
       r.payload->>'propai_estimated_completion'       AS estimated_completion,
       r.payload->>'project_description_en'            AS description,
       nullif(r.payload->>'latitude_prop_ai','')::float8  AS latitude,
       nullif(r.payload->>'longitude_prop_ai','')::float8 AS longitude,
       r.payload->>'last_inspection_date'              AS last_inspection_date,
       r.payload->>'inspection_report_url'             AS inspection_report_url,
       jsonb_array_length(coalesce(r.payload->'image_urls', '[]'::jsonb)) AS image_count,
       -- individual plot entries (e.g. 'Qpoint Liwan (Plot R093)') are not main projects; lists hide them
       (coalesce(r.payload->>'project_name_label', r.payload->>'project_name_en') ~* '\mplot\M') AS is_plot
FROM public.propai_raw r
LEFT JOIN public.propai_project_status st ON st.project_number = r.payload->>'project_number'
LEFT JOIN public.propai_project_prices pp ON pp.location_key = r.payload->>'prop-ai_location_key_project'
WHERE r.dataset = 'projects';

CREATE VIEW public.propai_projects AS
SELECT * FROM public.propai_projects_dld
UNION ALL
SELECT d.k, d.k, NULL::text, NULL::text, d.name, d.area, d.master_project, NULL::text, d.developer_group, initcap(pp.segment),
       CASE WHEN d.offplan THEN 'Under construction' ELSE 'Completed' END, NULL::text, NULL::text, NULL::text, NULL::text,
       NULL::numeric, NULL::date, NULL::date, NULL::date, NULL::text, NULL::text, NULL::text, NULL::double precision, NULL::double precision,
       NULL::text, NULL::text, 0, false
FROM public.propai_derived_projects d LEFT JOIN public.propai_project_prices pp ON pp.location_key = d.k;

-- One row per Dubai community (area) with sale and rent figures, for the communities directory.
-- Refresh after a big reload: REFRESH MATERIALIZED VIEW public.propai_area_stats;
DROP MATERIALIZED VIEW IF EXISTS public.propai_area_stats;
CREATE MATERIALIZED VIEW public.propai_area_stats AS
WITH s AS (
  SELECT area_name_prop_ai AS area, count(*)::int AS sales_total,
         count(*) FILTER (WHERE instance_date >= current_date - interval '12 months')::int AS sales_12m,
         (percentile_cont(0.5) WITHIN GROUP (ORDER BY sale_price) FILTER (WHERE instance_date >= current_date - interval '12 months'))::float8 AS median_price,
         (percentile_cont(0.5) WITHIN GROUP (ORDER BY price_per_sqft) FILTER (WHERE instance_date >= current_date - interval '12 months'))::float8 AS median_ppsf,
         max(instance_date) AS last_sale
  FROM public.propai_sales
  WHERE area_name_prop_ai IS NOT NULL AND deleted_at IS NULL AND coalesce(outlier_flag,'') <> 'True' AND instance_date <= current_date AND sale_price > 0
  GROUP BY 1),
r AS (
  SELECT area_name_prop_ai AS area, count(*)::int AS rent_contracts,
         (percentile_cont(0.5) WITHIN GROUP (ORDER BY annual_amount))::float8 AS median_rent
  FROM public.propai_rents
  WHERE area_name_prop_ai IS NOT NULL AND deleted_at IS NULL AND coalesce(outlier_flag,'') <> '1' AND contract_start_date <= current_date AND annual_amount > 0
  GROUP BY 1),
p AS (SELECT upper(area) AS area, count(*)::int AS projects FROM public.propai_projects WHERE NOT is_plot AND area IS NOT NULL GROUP BY 1),
cv AS (SELECT DISTINCT ON (upper(p.area)) upper(p.area) AS area, i.object_name AS cover_object FROM public.propai_projects p
       JOIN public.propai_project_images i ON i.project_key = p.key WHERE NOT p.is_plot ORDER BY upper(p.area), p.image_count DESC, i.position)
SELECT coalesce(s.area, r.area) AS area, coalesce(p.projects, 0) AS projects, coalesce(s.sales_total, 0) AS sales_total, coalesce(s.sales_12m, 0) AS sales_12m,
       s.median_price, s.median_ppsf, s.last_sale, coalesce(r.rent_contracts, 0) AS rent_contracts, r.median_rent, cv.cover_object
FROM s FULL JOIN r ON r.area = s.area LEFT JOIN p ON p.area = upper(coalesce(s.area, r.area)) LEFT JOIN cv ON cv.area = upper(coalesce(s.area, r.area));
CREATE UNIQUE INDEX propai_area_stats_area ON public.propai_area_stats(area);

COMMIT;
