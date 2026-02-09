"""
Prompt Schema - LLM-Friendly Column Aliases for Feature Interpretation

This module defines human-readable aliases for feature columns used in
LLM-based fraud detection. When generating prompts for language models,
these aliases make the feature names more interpretable both for the
model and for human reviewers of the analysis.

Structure:
    COL_ALIAS maps from internal column names to tuples containing:
    - display_name: LLM-friendly English name
    - description: Human-readable feature description (English)
    - unit_type: Data type or unit of measurement

Feature Categories:
    1. Ticketing Attributes: Station and route identifiers
    2. Transaction: Purchase and refund amounts
    3. Time Info: Timing features (lead time, hold time, etc.)
    4. Intentional Context: Day/hour patterns
    5. Route History: Historical usage patterns
    6. Round-trip Indicators: Commuter behavior signals
    7. Active Tickets: Currently held ticket features
    8. Fraud Signals: Suspicious pattern indicators
    9. Refund History: Recent refund behavior

Usage:
    >>> from prompt_schema import COL_ALIAS
    >>> alias, desc, unit = COL_ALIAS['hold_time']
    >>> print(f"{alias}: {desc}") 
    # ticket_holding_time: Time between purchase and refund
"""

# Original column name -> (display_name, English description, unit/type)
COL_ALIAS = {
    # =========================================================
    # Ticketing Attributes (Identifiers)
    # Used primarily for context, may be excluded from model input
    # =========================================================
    "dep_station_id": ("departure_station", "Departure station", "code"),
    "arr_station_id": ("arrival_station", "Arrival station", "code"),
    "route_id": ("route_key", "Route (departure-arrival pair)", "id/hash"),
    "train_id": ("train_number", "Train number", "id"),

    # =========================================================
    # Transaction Features
    # =========================================================
    "action_type": ("action", "Action type (purchase/refund)", "binary"),
    "seat_cnt": ("seats", "Number of seats", "count"),
    "buy_amt": ("purchase_amount", "Purchase amount", "KRW"),
    "refund_amt": ("refund_amount", "Refund amount", "KRW"),
    "cancel_fee": ("cancellation_fee", "Cancellation fee", "KRW"),
    "route_dist_km": ("trip_distance", "Travel distance", "km"),

    # =========================================================
    # Time Information
    # =========================================================
    "travel_time": ("trip_duration", "Trip travel time", "min"),
    "lead_time_buy": ("time_before_departure_at_purchase", "Time until departure at purchase", "min"),
    "lead_time_ref": ("time_before_departure_at_refund", "Time until departure at refund", "min"),
    "hold_time": ("ticket_holding_time", "Ticket holding duration (purchase to refund)", "min"),

    # =========================================================
    # Intentional Context
    # =========================================================
    "dep_dow": ("departure_day_of_week", "Departure day of week", "0~6"),
    "dep_hour": ("departure_hour", "Departure hour", "0~23"),

    # =========================================================
    # Route History (Pre-event window)
    # =========================================================
    "route_buy_cnt": ("route_purchases_in_window", "Same-route purchases in recent window", "count"),
    "fwd_dep_hour_median": ("forward_ticket_departure_hour_median", "Forward ticket departure hour (median)", "hour"),
    "fwd_dep_dow_median": ("forward_ticket_departure_dow_median", "Forward ticket departure day (median)", "dow"),

    # =========================================================
    # Commuter / Round-trip Indicators
    # =========================================================
    "rev_buy_cnt": ("reverse_route_purchases_in_window", "Reverse-route purchases in recent window", "count"),
    "rev_ratio": ("reverse_to_forward_purchase_ratio", "Reverse-to-forward purchase ratio", "ratio"),

    "completed_fwd_cnt": ("completed_forward_trips", "Completed forward trips count", "count"),
    "completed_fwd_dep_interval_median": ("completed_forward_departure_gap_median", "Forward departure interval (median)", "min"),
    "completed_fwd_dep_hour_median": ("completed_forward_departure_hour_median", "Forward departure hour (median)", "hour"),
    "completed_fwd_dep_dow_median": ("completed_forward_departure_dow_median", "Forward departure day (median)", "dow"),

    "completed_rev_cnt": ("completed_reverse_trips", "Completed reverse trips count", "count"),
    "completed_rev_dep_interval_median": ("completed_reverse_departure_gap_median", "Reverse departure interval (median)", "min"),
    "completed_rev_dep_hour_median": ("completed_reverse_departure_hour_median", "Reverse departure hour (median)", "hour"),
    "completed_rev_dep_dow_median": ("completed_reverse_departure_dow_median", "Reverse departure day (median)", "dow"),

    "unique_route_cnt": ("unique_routes_in_window", "Unique routes used in recent window", "count"),

    # =========================================================
    # Active Ticket Features (Return schedule indicators)
    # =========================================================
    "rev_dep_hour_median": ("active_reverse_departure_hour_median", "Active reverse ticket departure hour (median)", "hour"),
    "rev_dep_dow_median": ("active_reverse_departure_dow_median", "Active reverse ticket departure day (median)", "dow"),
    "rev_return_gap": ("min_gap_arrival_to_reverse_departure", "Minimum gap from arrival to reverse departure", "min"),

    # =========================================================
    # Fraud Signal Features
    # Red flags that indicate potential ticket fraud
    # =========================================================
    "overlap_cnt": ("overlapping_tickets_count", "Overlapping time period tickets count", "count"),
    "same_route_cnt": ("same_route_tickets_count", "Same-route active tickets count", "count"),
    "rev_route_cnt": ("reverse_route_tickets_count", "Reverse-route active tickets count", "count"),
    "repeat_interval": ("repeat_purchase_gap_median", "Same-route re-purchase interval (median)", "min"),
    "adj_seat_refund_flag": ("adjacent_seat_refund", "Adjacent seats refunded together flag", "0/1"),

    # =========================================================
    # Refund History
    # Recent refund behavior on same route
    # =========================================================
    "recent_ref_cnt": ("recent_refunds_count", "Recent same-route refunds count", "count"),
    "recent_ref_amt": ("recent_refunds_amount", "Recent same-route refund amount", "KRW"),
    "recent_ref_rate": ("recent_refund_ratio", "Recent same-route refund ratio", "ratio"),
}

