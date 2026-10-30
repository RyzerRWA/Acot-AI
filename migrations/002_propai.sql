-- Prop-AI API mirror: one raw row per record (jsonb) + resumable sync state.
-- Typed views/columns can be added later once the real field names are confirmed with --probe.
CREATE TABLE IF NOT EXISTS public.propai_raw (
    dataset    text        NOT NULL,          -- projects | rents | sales | brokers
    row_key    text        NOT NULL,
    payload    jsonb       NOT NULL,
    fetched_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (dataset, row_key)
);
CREATE TABLE IF NOT EXISTS public.propai_sync_state (
    job           text PRIMARY KEY,           -- dataset name, or '_budget'
    cursor        text,
    page          int         NOT NULL DEFAULT 1,
    since         timestamptz,                -- next run only asks for rows changed after this
    done          boolean     NOT NULL DEFAULT false,
    requests_used int         NOT NULL DEFAULT 0,
    updated_at    timestamptz NOT NULL DEFAULT now()
);
