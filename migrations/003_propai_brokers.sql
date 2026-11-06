-- Typed broker table, filled from propai_raw (dataset='brokers') by scripts/propai_sync.py.
-- Field names follow the Prop-AI /data/brokers/agents row. brn is the login identifier.
CREATE TABLE IF NOT EXISTS public.propai_brokers (
    brn                  text PRIMARY KEY,
    name_en              text,
    name_ar              text,
    agency_name          text,
    agency_name_ar       text,
    email                text,
    mobile               text,
    landline             text,
    status               text,
    photo_url            text,
    orn                  text,
    card_rank            text,
    office_rank          text,
    card_issue_date      date,
    card_expiry_date     date,
    office_issue_date    date,
    office_expiry_date   date,
    dld_card_found       boolean,
    dld_card_checked_at  timestamptz,
    annual_transactions  jsonb,
    project_transactions jsonb,
    award_history        jsonb,
    synced_at            timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS propai_brokers_email_idx  ON public.propai_brokers (lower(email));
CREATE INDEX IF NOT EXISTS propai_brokers_orn_idx    ON public.propai_brokers (orn);
CREATE INDEX IF NOT EXISTS propai_brokers_status_idx ON public.propai_brokers (status);
