-- Per Prop-AI project (matched on the project location key): average sale price over the last 24 months, with 12-month volume, previous-12-month volume and 12-month medians, plus which kind of
-- property it mostly holds (from all sales and rent records) so project lists can be split residential / office / commercial / land.
-- Refresh after a big reload: REFRESH MATERIALIZED VIEW public.propai_project_prices;
DROP MATERIALIZED VIEW IF EXISTS public.propai_project_prices;
CREATE MATERIALIZED VIEW public.propai_project_prices AS
WITH t AS (
  SELECT location_key_project AS k, property_type AS type FROM public.propai_sales WHERE location_key_project IS NOT NULL AND deleted_at IS NULL
  UNION ALL
  SELECT location_key_project, property_type FROM public.propai_rents WHERE location_key_project IS NOT NULL AND deleted_at IS NULL
), n AS (SELECT k, type, count(*) c FROM t WHERE type IS NOT NULL AND type <> 'Other' GROUP BY 1, 2),
seg AS (
  SELECT k, (array_agg(type ORDER BY c DESC))[1] AS main_type, (array_agg(type ORDER BY c DESC))[1:3] AS types,
         (array_agg(CASE WHEN type IN ('Apartment','Villa','Hotel Apartment','Stacked Townhouse','Stacked Townhouses') THEN 'residential'
                         WHEN type = 'Office' THEN 'office' WHEN type IN ('Shop','Showroom','Warehouse','Building') THEN 'commercial'
                         WHEN type = 'Land' THEN 'land' END ORDER BY c DESC))[1] AS segment
  FROM n GROUP BY k),
r AS (SELECT location_key_project AS k, count(*)::int AS rent_contracts FROM public.propai_rents WHERE location_key_project IS NOT NULL AND deleted_at IS NULL AND contract_start_date <= current_date GROUP BY 1),
p AS (
  SELECT location_key_project AS k, count(*)::int AS sales_count, avg(sale_price)::float8 AS avg_price, avg(price_per_sqft)::float8 AS avg_price_per_sqft,
         count(*) FILTER (WHERE instance_date >= current_date - interval '12 months')::int AS sales_12m,
         count(*) FILTER (WHERE instance_date < current_date - interval '12 months')::int AS sales_prev12m,
         (percentile_cont(0.5) WITHIN GROUP (ORDER BY sale_price) FILTER (WHERE instance_date >= current_date - interval '12 months'))::float8 AS median_price_12m,
         (percentile_cont(0.5) WITHIN GROUP (ORDER BY price_per_sqft) FILTER (WHERE instance_date >= current_date - interval '12 months'))::float8 AS median_ppsf_12m
  FROM public.propai_sales
  WHERE location_key_project IS NOT NULL AND instance_date >= current_date - interval '24 months' AND instance_date <= current_date AND sale_price > 0
        AND deleted_at IS NULL AND coalesce(outlier_flag,'') <> 'True'
  GROUP BY 1)
SELECT coalesce(seg.k, p.k) AS location_key, p.sales_count, p.avg_price, p.avg_price_per_sqft, p.sales_12m, p.sales_prev12m, p.median_price_12m, p.median_ppsf_12m, seg.main_type, seg.types, seg.segment, coalesce(r.rent_contracts, 0) AS rent_contracts
FROM seg FULL JOIN p ON p.k = seg.k LEFT JOIN r ON r.k = coalesce(seg.k, p.k);
CREATE UNIQUE INDEX propai_project_prices_key ON public.propai_project_prices(location_key);
