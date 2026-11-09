-- One row per Prop-AI project, flattened from the raw /data/projects/basic-details payload (propai_raw, dataset='projects').
-- image_urls are presigned links that expire after 1 hour, so only the count is exposed here; images need copying to our own storage.
CREATE OR REPLACE VIEW public.propai_projects AS
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
