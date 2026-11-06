-- Typed Prop-AI tables (field names from the live /data/sales and /data/rents responses).
-- Loaded by scripts/propai_sync.py with jsonb_populate_recordset, so JSON keys map to columns by name;
-- keys that are not listed here are dropped on purpose (millions of rows: keep them compact).
CREATE TABLE IF NOT EXISTS public.propai_sales (
    transaction_id text NOT NULL, source text NOT NULL,
    instance_date date, trans_group_en text, sale_type_id numeric,
    area_name_prop_ai text, master_project_prop_ai text, project_name_prop_ai text, building_name_prop_ai text,
    project_number text, property_type text, property_type_en text, property_sub_type_en text,
    floor text, pai_property_gid text, property_number text, bedrooms numeric, rooms_en text,
    development_status text, procedure_area numeric, procedure_area_sqft numeric, balcony_area numeric, built_up_area numeric,
    sale_price numeric, price_per_sqm numeric, price_per_sqft numeric, outlier_flag text,
    mortgage text, mortgage_percent numeric, num_times_sold numeric, last_sale_price numeric, last_sale_date date,
    change_last_sale numeric, change_last_sale_percentage numeric, developer_group text,
    location_key_area text, location_key_masterproject text, location_key_project text, location_key_building text,
    created_at timestamptz, updated_at timestamptz, deleted_at timestamptz,
    PRIMARY KEY (source, transaction_id)
);
CREATE INDEX IF NOT EXISTS propai_sales_building_idx ON public.propai_sales (location_key_building, instance_date DESC);
CREATE INDEX IF NOT EXISTS propai_sales_project_idx  ON public.propai_sales (location_key_project, instance_date DESC);
CREATE INDEX IF NOT EXISTS propai_sales_area_idx     ON public.propai_sales (location_key_area, instance_date DESC);
CREATE INDEX IF NOT EXISTS propai_sales_date_idx     ON public.propai_sales (instance_date DESC);

CREATE TABLE IF NOT EXISTS public.propai_rents (
    pai_contract_id text NOT NULL, unit_seq int NOT NULL, pai_contract_number text,
    contract_registration_date date, contract_start_date date, contract_end_date date, contract_months numeric,
    contract_type text, status text,
    area_name_prop_ai text, master_project_prop_ai text, project_name_prop_ai text, building_name_prop_ai text,
    project_number text, property_type text, floor text, pai_property_gid text, property_number text,
    bedrooms numeric, development_status text, actual_area numeric, actual_area_sqft numeric, balcony_area numeric,
    annual_amount numeric, rent_per_sqm numeric, rent_per_sqft numeric, annualization_source text, outlier_flag text,
    contract_amount numeric, total_annual_amount numeric, is_multi_unit boolean, units_in_contract numeric,
    allocation_method text, allocated_annual_amount numeric, allocated_contract_amount numeric,
    bulk_registration boolean, bulk_group_size numeric, developer_group text,
    location_key_area text, location_key_masterproject text, location_key_project text, location_key_building text,
    created_at timestamptz, updated_at timestamptz, deleted_at timestamptz,
    PRIMARY KEY (pai_contract_id, unit_seq)
);
CREATE INDEX IF NOT EXISTS propai_rents_building_idx ON public.propai_rents (location_key_building, contract_start_date DESC);
CREATE INDEX IF NOT EXISTS propai_rents_project_idx  ON public.propai_rents (location_key_project, contract_start_date DESC);
CREATE INDEX IF NOT EXISTS propai_rents_area_idx     ON public.propai_rents (location_key_area, contract_start_date DESC);
