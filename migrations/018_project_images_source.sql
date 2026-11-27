-- Where each stored project image came from: 'propai' (scripts/propai_project_images.py) or 'web'
-- (scripts/web_project_images.py, the developer's official site). source_url is the page it was found on.
ALTER TABLE public.propai_project_images
  ADD COLUMN IF NOT EXISTS source     text NOT NULL DEFAULT 'propai',
  ADD COLUMN IF NOT EXISTS source_url text;
