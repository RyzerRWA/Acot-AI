-- Chat/account tables and functions for the ACOT AI service (schema acot).
-- Extracted verbatim from the previous Supabase database. Safe to re-run.

create table if not exists acot.subscription_plans (
  plan_name text primary key,
  monthly_token_limit integer not null
);

create table if not exists acot.monthly_token_usage (
  user_id text not null references acot.users(id) on delete cascade,
  gmail text not null,
  plan text not null,
  tokens_used bigint not null default 0,
  tokens_remaining bigint not null default 0,
  period_start date not null,
  updated_at timestamptz not null default now(),
  primary key (user_id, period_start)
);

create table if not exists acot.account_intelligence (
  user_id text primary key references acot.users(id) on delete cascade,
  intelligence jsonb not null default '{}'::jsonb,
  question_count integer not null default 0,
  conversation_count integer not null default 0,
  analyzer_version text not null default '1.0',
  updated_at timestamptz not null default now()
);

CREATE OR REPLACE FUNCTION public.change_subscription_plan(p_user_id text, p_plan text, p_billing_cycle text DEFAULT 'monthly'::text)
 RETURNS jsonb
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public', 'acot'
AS $function$
DECLARE
    v_plan text := lower(trim(p_plan));
    v_period date := date_trunc('month', now())::date;
    v_limit integer;
BEGIN
    IF v_plan NOT IN ('free', 'go', 'pro') THEN
        RAISE EXCEPTION 'Invalid plan. Allowed: free, go, pro';
    END IF;

    SELECT monthly_token_limit
    INTO v_limit
    FROM acot.subscription_plans
    WHERE plan_name = v_plan;

    IF v_limit IS NULL THEN
        RAISE EXCEPTION 'Plan configuration not found: %', v_plan;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM acot.users
        WHERE id = p_user_id
    ) THEN
        RAISE EXCEPTION 'User not found: %', p_user_id;
    END IF;

    UPDATE acot.users
    SET
        plan = v_plan,
        billing_cycle = p_billing_cycle
    WHERE id = p_user_id;

    INSERT INTO acot.monthly_token_usage (
        user_id,
        period_start,
        tokens_used,
        updated_at
    )
    VALUES (
        p_user_id,
        v_period,
        0,
        now()
    )
    ON CONFLICT (user_id, period_start)
    DO UPDATE SET
        tokens_used = 0,
        updated_at = now();

    RETURN jsonb_build_object(
        'success', true,
        'user_id', p_user_id,
        'plan', v_plan,
        'billing_cycle', p_billing_cycle,
        'monthly_limit', v_limit,
        'tokens_used', 0,
        'tokens_remaining', v_limit,
        'period_start', v_period
    );
END;
$function$;

CREATE OR REPLACE FUNCTION public.consume_monthly_tokens(p_user_id text, p_tokens integer)
 RETURNS jsonb
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO ''
AS $function$
DECLARE
    v_plan text;
    v_limit integer;
    v_period date := date_trunc('month', now())::date;
    v_used bigint;
BEGIN
    SELECT COALESCE(NULLIF(lower(plan), ''), 'free')
    INTO v_plan
    FROM acot.users
    WHERE id = p_user_id;

    IF v_plan IS NULL THEN
        RAISE EXCEPTION 'User not found';
    END IF;

    SELECT monthly_token_limit
    INTO v_limit
    FROM acot.subscription_plans
    WHERE plan_name = v_plan;

    IF v_limit IS NULL THEN
        RAISE EXCEPTION 'Unknown subscription plan: %', v_plan;
    END IF;

    INSERT INTO acot.monthly_token_usage (
        user_id, period_start, tokens_used
    )
    VALUES (
        p_user_id, v_period, 0
    )
    ON CONFLICT (user_id, period_start)
    DO NOTHING;

    SELECT tokens_used
    INTO v_used
    FROM acot.monthly_token_usage
    WHERE user_id = p_user_id
      AND period_start = v_period
    FOR UPDATE;

    IF p_tokens > 0 AND v_used + p_tokens > v_limit THEN
        RETURN jsonb_build_object(
            'allowed', false,
            'plan', v_plan,
            'monthly_limit', v_limit,
            'tokens_used', v_used,
            'tokens_remaining', GREATEST(v_limit - v_used, 0),
            'period_start', v_period
        );
    END IF;

    IF p_tokens > 0 THEN
        UPDATE acot.monthly_token_usage
        SET tokens_used = tokens_used + p_tokens,
            updated_at = now()
        WHERE user_id = p_user_id
          AND period_start = v_period;

        v_used := v_used + p_tokens;
    END IF;

    RETURN jsonb_build_object(
        'allowed', true,
        'plan', v_plan,
        'monthly_limit', v_limit,
        'tokens_used', v_used,
        'tokens_remaining', GREATEST(v_limit - v_used, 0),
        'period_start', v_period
    );
END;
$function$;

CREATE OR REPLACE FUNCTION public.get_account_intelligence(p_user_id text)
 RETURNS jsonb
 LANGUAGE sql
 STABLE SECURITY DEFINER
 SET search_path TO ''
AS $function$
    SELECT COALESCE(
        (
            SELECT jsonb_build_object(
                'user_id', user_id,
                'intelligence', intelligence,
                'question_count', question_count,
                'conversation_count', conversation_count,
                'analyzer_version', analyzer_version,
                'updated_at', updated_at
            )
            FROM acot.account_intelligence
            WHERE user_id = p_user_id
        ),
        '{}'::jsonb
    );
$function$;

CREATE OR REPLACE FUNCTION public.get_user_questions(p_user_id text)
 RETURNS TABLE(conversation_id text, message_id text, question text, created_at timestamp with time zone)
 LANGUAGE sql
 STABLE SECURITY DEFINER
 SET search_path TO ''
AS $function$
    SELECT
        m.conversation_id,
        m.id,
        m.question,
        m.created_at
    FROM acot.conversations c
    JOIN acot.messages m
        ON m.conversation_id = c.id
    WHERE c.user_id = p_user_id
      AND m.role = 'user'
      AND m.question IS NOT NULL
      AND trim(m.question) <> ''
    ORDER BY m.created_at;
$function$;

CREATE OR REPLACE FUNCTION public.save_account_intelligence(p_user_id text, p_intelligence jsonb, p_question_count integer, p_conversation_count integer)
 RETURNS void
 LANGUAGE sql
 SECURITY DEFINER
 SET search_path TO ''
AS $function$
    INSERT INTO acot.account_intelligence (
        user_id,
        intelligence,
        question_count,
        conversation_count,
        analyzer_version,
        updated_at
    )
    VALUES (
        p_user_id,
        p_intelligence,
        p_question_count,
        p_conversation_count,
        '1.0',
        now()
    )
    ON CONFLICT (user_id)
    DO UPDATE SET
        intelligence = EXCLUDED.intelligence,
        question_count = EXCLUDED.question_count,
        conversation_count = EXCLUDED.conversation_count,
        analyzer_version = EXCLUDED.analyzer_version,
        updated_at = now();
$function$;
