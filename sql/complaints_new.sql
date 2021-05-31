-- New query after Utkarsh implementation of recording the language portal in the moment of feedback submition
select
        customer_care_id
         ,cce.timestamp
         , json_extract_path_text(cce.args, 'feedback') as message
         , json_extract_path_text(cce.args, 'reason') as reason
         , json_extract_path_text(cce.args, 'lang') as language
    from customer_care_flow_events cce
    where cce.action = 'feedback_success'
      and  timestamp >= '2021-05-28'
      and json_extract_path_text(cce.args, 'lang') != ''
      and json_extract_path_text(cce.args, 'feedback') != ''