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
