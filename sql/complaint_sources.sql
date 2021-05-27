with relations as (
    select
           json_extract_path_text(args, 'sub', 'rockmanId') as rockman_id
         , customer_care_id
    from customer_care_flow_events
    where action = 'unsub_success'  -- use the review submission_review success
)

select r.customer_care_id
     , rockman_id
     , google_placement_name
     , google_tid
     , country_code
     , json_extract_path_text(query_string, 'g_adgroupid') as vertical
     , json_extract_path_text(query_string, 'g_creative') as banner
from relations r left join user_subscriptions us using (rockman_id)