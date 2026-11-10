-- DLD project status per project number (Prop-AI /data/projects/status): progress, inspections, developer and management-company contacts.
CREATE TABLE IF NOT EXISTS public.propai_project_status (
  project_number text PRIMARY KEY,
  payload        jsonb NOT NULL,
  fetched_at     timestamptz NOT NULL DEFAULT now()
);
