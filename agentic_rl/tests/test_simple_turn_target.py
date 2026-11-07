from copy import deepcopy
import pytest
from tau2_agentic_rl.reward import simple_turn as rule


def frozen():
 return rule.build_turn_inputs([
  {'role':'system','content':'Use only supplied facts.'},
  {'role':'user','content':'The balance is $250.'},
  {'role':'assistant','turn_idx':1,'content':'The balance is $250.'},
  {'role':'user','content':'Confirm the balance.'},
  {'role':'assistant','turn_idx':2,'content':'The balance is $500.','raw_generated_text':'The balance is $500.<|im_end|>'},
  {'role':'user','content':'hidden future marker'},
 ],'Use only supplied facts.',2)


def valid(request):
 return {'turn_index':request['turn_index'],
         'target_message_index':request['target_message_index'],
         'target_generation_sha256':request['target_generation_sha256'],
         'target_quote':request['target_generation_text'],
         'bad_turn':True,'correct_refusal':False,'reason':'The current generation claims $500 instead of visible $250.',
         'evidence_message_indices':[1,request['target_message_index']],
         'refusal_requests':[],'revoked_refusals':[]}


def test_current_generation_is_explicit_and_metadata_free_without_mutating_source():
 inputs=frozen();before=deepcopy(inputs);req=rule.turn_request(inputs,1,[])
 assert 'current_generation' in req and 'history' in req
 assert 'messages' not in req
 assert req['target_message_index']==4
 assert req['current_generation']['message_index']==4
 assert req['current_generation']['message']['content']=='The balance is $500.'
 assert [m['message_index'] for m in req['history']]==[0,1,2,3]
 assert all('turn_idx' not in item['message'] for item in req['history']+[req['current_generation']])
 assert 'hidden future marker' not in str(req)
 assert inputs==before


def test_legacy_response_that_assesses_previous_generation_is_rejected():
 req=rule.turn_request(frozen(),1,[])
 old={'turn_index':1,'bad_turn':False,'correct_refusal':False,'reason':'The assistant states $250 correctly.',
      'evidence_message_indices':[1,2],'refusal_requests':[],'revoked_refusals':[]}
 with pytest.raises(ValueError):rule.validate_decision(old,req)


def test_valid_target_binding_roundtrips():
 req=rule.turn_request(frozen(),1,[]);d=valid(req)
 assert rule.validate_decision(d,req)==d


@pytest.mark.parametrize('field,value',[
 ('target_message_index',2),('target_generation_sha256','0'*64),
 ('target_quote','The balance is $250.'),('target_quote',''),
 ('evidence_message_indices',[1,2]),('evidence_message_indices',[4,5]),
])
def test_wrong_target_binding_quote_or_evidence_is_rejected(field,value):
 req=rule.turn_request(frozen(),1,[]);d=valid(req);d[field]=value
 with pytest.raises(ValueError):rule.validate_decision(d,req)


def test_neutral_decision_also_has_to_address_current_generation():
 req=rule.turn_request(frozen(),1,[]);d=valid(req);d['bad_turn']=False;d['evidence_message_indices']=[1,2]
 with pytest.raises(ValueError):rule.validate_decision(d,req)


def test_legacy_labels_cannot_silently_enter_training():
 legacy={'version':'simple-turn-v1','turns':[{'turn_index':0,'bad_turn':False,'eligible_correct_refusal':False}]}
 with pytest.raises(ValueError,match='rescor|target|Judge'):rule.validate_labels(legacy,1)


@pytest.mark.parametrize('field,value',[
 ('target_message_index',3),('target_generation_sha256','f'*64),
 ('target_generation_text','different generation'),
])
def test_request_target_tampering_is_rejected(field,value):
 req=rule.turn_request(frozen(),1,[]);d=valid(req);req[field]=value
 with pytest.raises(ValueError):rule.validate_decision(d,req)


def test_tool_generation_retains_all_raw_arguments_and_hides_future_result():
 raw='<tool_call>{"name":"update_reservation_flights","arguments":{"reservation_id":"AB12CD","payment_id":"gift_card_123"}}</tool_call>'
 messages=[{'role':'user','content':'Please proceed.'},
           {'role':'assistant','turn_idx':1,'raw_generated_text':raw,'tool_calls':[{'function':{'name':'update_reservation_flights','arguments':'{"reservation_id":"AB12CD","payment_id":"gift_card_123"}'}}]},
           {'role':'tool','content':'future hidden tool result'}]
 inputs=rule.build_turn_inputs(messages,'bound policy',1);before=deepcopy(inputs)
 req=rule.turn_request(inputs,0,[])
 assert req['target_generation_text']==raw
 assert req['current_generation']['message']['tool_calls']==messages[1]['tool_calls']
 assert 'future hidden tool result' not in str(req)
 assert inputs==before
