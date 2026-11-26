-- One row per master project (Dubai World Central, Majan, Jumeirah Village Circle ...): the community names people use,
-- which are often not DLD area names. Same figures as propai_area_stats, so a community page can open for either.
-- Refresh after a big reload: REFRESH MATERIALIZED VIEW public.propai_master_stats;
DROP MATERIALIZED VIEW IF EXISTS public.propai_master_stats;
CREATE MATERIALIZED VIEW public.propai_master_stats AS
WITH s AS (
  SELECT upper(btrim(regexp_replace(master_project_prop_ai, '\s*\([^)]*\)', '', 'g'))) AS k, (array_agg(master_project_prop_ai ORDER BY instance_date DESC))[1] AS name,
         count(*)::int AS sales_total,
         count(*) FILTER (WHERE instance_date >= current_date - interval '12 months')::int AS sales_12m,
         (percentile_cont(0.5) WITHIN GROUP (ORDER BY sale_price) FILTER (WHERE instance_date >= current_date - interval '12 months'))::float8 AS median_price,
         (percentile_cont(0.5) WITHIN GROUP (ORDER BY price_per_sqft) FILTER (WHERE instance_date >= current_date - interval '12 months'))::float8 AS median_ppsf,
         max(instance_date) AS last_sale
  FROM public.propai_sales
  WHERE coalesce(btrim(master_project_prop_ai), '') <> '' AND deleted_at IS NULL AND coalesce(outlier_flag,'') <> 'True'
        AND instance_date <= current_date AND sale_price > 0
  GROUP BY 1),
r AS (
  SELECT upper(btrim(regexp_replace(master_project_prop_ai, '\s*\([^)]*\)', '', 'g'))) AS k, count(*)::int AS rent_contracts,
         (percentile_cont(0.5) WITHIN GROUP (ORDER BY annual_amount))::float8 AS median_rent
  FROM public.propai_rents
  WHERE coalesce(btrim(master_project_prop_ai), '') <> '' AND deleted_at IS NULL AND coalesce(outlier_flag,'') <> '1'
        AND contract_start_date <= current_date AND annual_amount > 0
  GROUP BY 1),
p AS (SELECT upper(btrim(regexp_replace(master_project, '\s*\([^)]*\)', '', 'g'))) AS k, count(*)::int AS projects, (array_agg(btrim(regexp_replace(master_project, '\s*\([^)]*\)', '', 'g')) ORDER BY image_count DESC))[1] AS label FROM public.propai_projects
      WHERE NOT is_plot AND coalesce(btrim(master_project), '') <> '' GROUP BY 1),
cv AS (SELECT DISTINCT ON (upper(btrim(regexp_replace(p.master_project, '\s*\([^)]*\)', '', 'g')))) upper(btrim(regexp_replace(p.master_project, '\s*\([^)]*\)', '', 'g'))) AS k, i.object_name AS cover_object
       FROM public.propai_projects p JOIN public.propai_project_images i ON i.project_key = p.key
       WHERE NOT p.is_plot AND coalesce(btrim(p.master_project), '') <> '' ORDER BY upper(btrim(regexp_replace(p.master_project, '\s*\([^)]*\)', '', 'g'))), p.image_count DESC, i.position)
SELECT s.k, coalesce(p.label, initcap(lower(s.name))) AS name, coalesce(p.projects, 0) AS projects, s.sales_total, s.sales_12m, s.median_price, s.median_ppsf, s.last_sale,
       coalesce(r.rent_contracts, 0) AS rent_contracts, r.median_rent, cv.cover_object
FROM s LEFT JOIN r ON r.k = s.k LEFT JOIN p ON p.k = s.k LEFT JOIN cv ON cv.k = s.k;
CREATE UNIQUE INDEX propai_master_stats_k ON public.propai_master_stats(k);
