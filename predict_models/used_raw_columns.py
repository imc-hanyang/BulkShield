"""
Feature Column Definitions - Raw Column Names for Model Input

This module defines the list of feature columns extracted from the preprocessed
transaction data and used as inputs to the fraud detection models. These columns
represent behavioral features engineered from raw ticketing events.

The columns are used by:
    - LazyDataset and other Dataset classes for CSV parsing
    - Model input layer dimension calculations
    - Feature importance analysis and interpretation

Feature Categories (37 total features):
    - Identifiers (4): Station IDs, route ID, train ID
    - Transaction (6): Action type, seats, amounts, distance, travel time
    - Time Features (5): Lead times, hold time, day/hour
    - Route History (6): Purchase counts, departure patterns
    - Round-trip (9): Completed trip statistics
    - Active Tickets (3): Reverse ticket features
    - Fraud Signals (5): Overlap, same-route, adjacent seat flags
    - Refund History (3): Recent refund behavior

Note:
    Columns like 'user_id', 'anchor_day', 'anchor_time', 'timestamp' are
    excluded from model input but may be used for data organization.
"""

# Feature columns used as model inputs
# Excludes metadata columns: user_id, anchor_day, anchor_time, timestamp
columns = [
    # Ticketing Identifiers
    "dep_station_id", "arr_station_id", "route_id", "train_id",
    
    # Transaction Features
    "action_type", "seat_cnt", "buy_amt", "refund_amt", "cancel_fee", "route_dist_km",
    
    # Time Features
    "travel_time", "lead_time_buy", "lead_time_ref", "hold_time", "dep_dow", "dep_hour",
    
    # Route Purchase History
    "route_buy_cnt", "fwd_dep_hour_median", "fwd_dep_dow_median", "rev_buy_cnt", "rev_ratio",
    
    # Route Usage Diversity
    "unique_route_cnt",
    
    # Active Reverse Ticket Features
    "rev_dep_hour_median", "rev_dep_dow_median", "rev_return_gap",
    
    # Fraud Signal Features
    "overlap_cnt", "same_route_cnt", "rev_route_cnt", "repeat_interval", "adj_seat_refund_flag",
    
    # Recent Refund History
    "recent_ref_cnt", "recent_ref_amt", "recent_ref_rate",
]