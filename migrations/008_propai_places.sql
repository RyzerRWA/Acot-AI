-- Lookup of Prop-AI places (building / project / master project / area) built from the stored sales, so a listing's
-- location text can be matched to a Prop-AI location key. norm = name upper-cased with everything but A-Z0-9 removed.
-- Refresh by re-running the insert (truncate first) after a big sales load.
CREATE TABLE IF NOT EXISTS public.propai_places (
    level        text   NOT NULL,   -- building | project | master | area
    norm         text   NOT NULL,
    location_key text   NOT NULL,
    label        text   NOT NULL,
    area         text,
    master       text,
    sales_count  int    NOT NULL,
    PRIMARY KEY (level, location_key)
);
CREATE INDEX IF NOT EXISTS propai_places_norm_idx ON public.propai_places (norm);

TRUNCATE public.propai_places;
INSERT INTO public.propai_places (level, norm, location_key, label, area, master, sales_count)
SELECT 'building', regexp_replace(upper(max(building_name_prop_ai)), '[^A-Z0-9]', '', 'g'), location_key_building,
       max(building_name_prop_ai), max(area_name_prop_ai), max(master_project_prop_ai), count(*)
FROM public.propai_sales WHERE location_key_building IS NOT NULL AND building_name_prop_ai <> '' GROUP BY location_key_building
UNION ALL
SELECT 'project', regexp_replace(upper(max(project_name_prop_ai)), '[^A-Z0-9]', '', 'g'), location_key_project,
       max(project_name_prop_ai), max(area_name_prop_ai), max(master_project_prop_ai), count(*)
FROM public.propai_sales WHERE location_key_project IS NOT NULL AND project_name_prop_ai <> '' GROUP BY location_key_project
UNION ALL
SELECT 'master', regexp_replace(upper(max(master_project_prop_ai)), '[^A-Z0-9]', '', 'g'), location_key_masterproject,
       max(master_project_prop_ai), max(area_name_prop_ai), max(master_project_prop_ai), count(*)
FROM public.propai_sales WHERE location_key_masterproject IS NOT NULL AND master_project_prop_ai <> '' GROUP BY location_key_masterproject
UNION ALL
SELECT 'area', regexp_replace(upper(max(area_name_prop_ai)), '[^A-Z0-9]', '', 'g'), location_key_area,
       max(area_name_prop_ai), max(area_name_prop_ai), NULL, count(*)
FROM public.propai_sales WHERE location_key_area IS NOT NULL AND area_name_prop_ai <> '' GROUP BY location_key_area;

CREATE INDEX IF NOT EXISTS propai_sales_master_idx ON public.propai_sales (location_key_masterproject, instance_date DESC);
CREATE INDEX IF NOT EXISTS propai_rents_master_idx ON public.propai_rents (location_key_masterproject, contract_start_date DESC);
