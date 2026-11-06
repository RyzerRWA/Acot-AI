-- Off-plan payment plans and property types per project (Prop-AI /data/projects/details/:location_key).
CREATE TABLE IF NOT EXISTS public.propai_project_plans (
  location_key   text PRIMARY KEY,
  payment_plans  jsonb,
  property_types text,
  fetched_at     timestamptz NOT NULL DEFAULT now()
);
