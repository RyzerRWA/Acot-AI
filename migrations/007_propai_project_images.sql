-- Project images copied from Prop-AI (presigned, 1-hour links) into our Google Cloud photos bucket.
-- object_name is the path inside the bucket; the public URL is <bucket base URL>/<object_name>.
CREATE TABLE IF NOT EXISTS public.propai_project_images (
    project_key  text NOT NULL,     -- propai_projects.key
    position     int  NOT NULL,     -- 0 = cover
    object_name  text NOT NULL,
    bytes        int,
    content_type text,
    copied_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (project_key, position)
);
