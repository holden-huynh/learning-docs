# Data

## <span style=color:yellow> 1. analytics_298764602 </span>
* from LS


## <span style=color:yellow> 2. analytics_179070117 </span>
### <span style=color:green> events_
* raw events
    * event_date
    * event_timestamp
    * event_name
    * event_params
        * key
        * value
            * string_value
            * int_value
            * float_value
            * double_value
    * user_properties
        * same as event_params
    * user_id
    * user_pseudo_id

### <span style=color:green> events_intraday_
* same as __events__

### <span style=color:green> daily raw
* mr (+ inkr?) (from __2/7/2018__ to __4/8/2021__)
* pre-processed events:
    * date / time
    * event_name
    * params
    * user_id / app_instance


## <span style=color:yellow> 3. unified_analytics </span>
### <span style=color:green> daily_raw
* inkr only (from 14/4/2020 up to __NOW__)
* pre-processed: matched user_ids
* events:
    * event_timestamp
    * event_name
    * event_params
    * user_properties
    * user_id / app_instance
    * first_open_timestamp?
    * app platform / versions
    * is_premium / is_subscribed?
    * language / country
    * ...

### <span style=color:green> daily_ecommerce_raw

### <span style=color:green> daily_acc_app_id

### <span style=color:green> daily_app_instance_active
* user_engagement, including app_remove

### <span style=color:green> daily_active_user
* user_engagement, excluding app_remove

### <span style=color:green> daily_user_engagement
* user engagement + timespent

### <span style=color:green> daily_first_open

### <span style=color:green> daily_first_visit

### <span style=color:green> daily_new_user_sign_up

### <span style=color:green> daily_chapter_read
* unlocked_method?
* is_locked?
* is_pass?

### <span style=color:green> daily_page_view: GA?

### <span style=color:green> uid_to_ignore_from_report


## <span style=color:yellow> 4. manga-rock-9cfc9.analytics_152146348
### daily_raw: 
* not used?


## <span style=color:yellow> 5. manga-rock-9cfc9.unified_analytics
### <span style=color:green> daily_raw
* countly + analytics_179070117.daily_raw (inkr + mr)


## <span style=color:yellow> 6. content_reporting


## <span style=color:yellow> 7. content_stats


## <span style=color:yellow> 8. business_key_health_metrics
### <span style=color:green> audience_weekly_retention

### <span style=color:green> daily_revenue
    
### <span style=color:green> daily_retention
    
### <span style=color:green> customer_transactions
    
### <span style=color:green> daily_customer_transactions

### <span style=color:green> daily_engaged_reader_active_status
    

## <span style=color:yellow> 9. 



---
# Pipelines
## <span style=color:yellow> 1. inkr_firebase_pipeline
### preprocessing
* daily_acc_app_id
    * extract user_id, app_instance_id from user_properties
    * 
* prepare_audience
* matching_users

### event_processing
* daily_raw
* daily_*

### event_aggregation
* manga_chapter_read
* manga_page_view
* user_first_read_title

### downstream

## <span style=color:yellow> 2. inkr_firebase_intraday
### preprocessing

### event_processing

### event_aggregation

### downstream


## <span style=color:yellow> 2. inkr_payment_dump
* dump 3 tables:
    * payment_transaction
    * payment_coin_purchase_item
    * payment_customer
* calculate:    
    * chapter_transaction
    * coin_transaction
    * subscription_transaction

## <span style=color:yellow> 3. inkr_subscription

## <span style=color:yellow> 4. recsys
* daily_read_acc.sql:
    - process data to read_agg_20161201_today
    - copy to read_agg_all
* run action_agg.sql:
    - process
        - read_agg_all: aggregated read for MR & INKR
        - fav_mr_data_agg_20161201_20200623
    - write to agg_all