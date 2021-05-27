with most_used_lang as (
    select customer_care_id,
           portal_language,
           first_value(lower(portal_language)) over (partition by customer_care_id order by count(*) desc rows between unbounded preceding and current row) as most_used_portal_language
    from customer_care_pageviews
    group by 1, 2
) ,

cleaned_used_lang as(
    select customer_care_id,
           most_used_portal_language
    from most_used_lang
    group by customer_care_id, most_used_portal_language
)

select cce.customer_care_id as customer_care_id,
       cce.timestamp,
       json_extract_path_text(cce.args, 'feedback') as message,
       lang.most_used_portal_language as portal_language
from customer_care_flow_events cce
    left join cleaned_used_lang lang
        on cce.customer_care_id = lang.customer_care_id
where
      cce.action = 'feedback_success'
  and json_extract_path_text(cce.args, 'feedback') != ''
order by 1
