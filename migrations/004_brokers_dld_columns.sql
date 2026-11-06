-- Extra columns for brokers loaded straight from the DLD public broker search (scripts/dld_brokers_sync.py).
ALTER TABLE public.propai_brokers
    ADD COLUMN IF NOT EXISTS card_rank_id   int,
    ADD COLUMN IF NOT EXISTS office_rank_id int,
    ADD COLUMN IF NOT EXISTS awards_count   int,
    ADD COLUMN IF NOT EXISTS office_logo    text,
    ADD COLUMN IF NOT EXISTS license_number text,
    ADD COLUMN IF NOT EXISTS source         text;
