-- Full Prop-AI project details per project (/data/projects/details/:location_key):
-- amenities_N, highlight_N, value_proposition_N, layouts (starting prices), payment plans,
-- brochure/floorplan links, executive summary. The chat reads amenities and features from here.
CREATE TABLE IF NOT EXISTS public.propai_project_details (
  location_key text PRIMARY KEY,
  payload      jsonb,
  fetched_at   timestamptz NOT NULL DEFAULT now()
);
-- Amenities from the developer's official site, for projects Prop-AI has none for (scripts/web_project_amenities.py).
-- Separate columns so a Prop-AI refresh of payload never wipes them.
ALTER TABLE public.propai_project_details
  ADD COLUMN IF NOT EXISTS web_amenities  jsonb,
  ADD COLUMN IF NOT EXISTS official_url   text,
  ADD COLUMN IF NOT EXISTS web_checked_at timestamptz;
