import json
import logging
import os
import functools
import time
from typing import List, Dict, Iterable
from multiprocessing import Pool, TimeoutError
from pathlib import Path

import dill
from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, Tokenizer, TokenIndexer, Field, Instance, Token
from allennlp.data.dataset_readers import MultiprocessDatasetReader
from allennlp.data.fields import TextField, ProductionRuleField, ListField, IndexField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter, BertBasicWordSplitter
from overrides import overrides
from spacy.symbols import ORTH, LEMMA

from dataset_readers.dataset_util.spider_utils import fix_number_value, disambiguate_items
from dataset_readers.fields.knowledge_graph_field import SpiderKnowledgeGraphField
from semparse.contexts.spider_db_context import SpiderDBContext
from semparse.worlds.spider_world import SpiderWorld

logger = logging.getLogger(__name__)


@DatasetReader.register("spider")
class SpiderDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 question_token_indexers: Dict[str, TokenIndexer] = None,
                 keep_if_unparsable: bool = True,
                 tables_file: str = None,
                 dataset_path: str = 'dataset/database',
                 load_cache: bool = True,
                 save_cache: bool = True,
                 loading_limit = -1):
        super().__init__(lazy=lazy)

        self.cache_data('cache-bert-fixgr')

        # default spacy tokenizer splits the common token 'id' to ['i', 'd'], we here write a manual fix for that
        spacy_tokenizer = SpacyWordSplitter(pos_tags=True)
        spacy_tokenizer.spacy.tokenizer.add_special_case(u'id', [{ORTH: u'id', LEMMA: u'id'}])
        self._tokenizer = WordTokenizer(spacy_tokenizer)

        self._utterance_token_indexers = question_token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._keep_if_unparsable = keep_if_unparsable

        self._tables_file = tables_file
        self._dataset_path = dataset_path

        self._loading_limit = loading_limit
    
    @overrides
    def _instances_from_cache_file(self, cache_filename: str) -> Iterable[Instance]:
        print('read instance from', cache_filename)
        with open(cache_filename, 'rb') as cache_file:
            instances = dill.load(cache_file)
            for instance in instances:
                yield instance

    @overrides
    def _instances_to_cache_file(self, cache_filename, instances) -> None:
        print('write instance to', cache_filename)
        with open(cache_filename, 'wb') as cache_file:
            dill.dump(instances, cache_file)

    @overrides
    def _read(self, file_path: str):
        print(file_path)
        cnt = 0
        with open(file_path, "r") as data_file:
            json_obj = json.load(data_file)

            with Pool(processes=12) as pool:
                ret = list(pool.map(self.process_ex, json_obj))
            ret = [i for i in ret if i is not None]
            print("Size:", len(ret))
            return ret
    
    def process_ex(self, ex):
        query_tokens = None
        if 'query_toks' in ex:
            # we only have 'query_toks' in example for training/dev sets

            # fix for examples: we want to use the 'query_toks_no_value' field of the example which anonymizes
            # values. However, it also anonymizes numbers (e.g. LIMIT 3 -> LIMIT 'value', which is not good
            # since the official evaluator does expect a number and not a value
            ex = fix_number_value(ex)

            # we want the query tokens to be non-ambiguous (i.e. know for each column the table it belongs to,
            # and for each table alias its explicit name)
            # we thus remove all aliases and make changes such as:
            # 'name' -> 'singer@name',
            # 'singer AS T1' -> 'singer',
            # 'T1.name' -> 'singer@name'
            try:
                query_tokens = disambiguate_items(ex['db_id'], ex['query_toks_no_value'],
                                                        self._tables_file, allow_aliases=False)
            except Exception as e:
                # there are two examples in the train set that are wrongly formatted, skip them
                print(f"error with {ex['query']}")
                print(e)

        ins = self.text_to_instance(
            utterance=ex['question'],
            db_id=ex['db_id'],
            sql=query_tokens)

        if ins is not None:
            return ins

    def text_to_instance(self,
                         utterance: str,
                         db_id: str,
                         sql: List[str] = None):
        fields: Dict[str, Field] = {}

        db_context = SpiderDBContext(db_id, utterance, tokenizer=self._tokenizer,
                                     tables_file=self._tables_file, dataset_path=self._dataset_path)
        table_field = SpiderKnowledgeGraphField(db_context.knowledge_graph,
                                                db_context.tokenized_utterance,
                                                {},
                                                entity_tokens=db_context.entity_tokens,
                                                include_in_vocab=False,  # TODO: self._use_table_for_vocab,
                                                max_table_tokens=None)  # self._max_table_tokens)


        combined_tokens = [] + db_context.tokenized_utterance
        entity_token_map = dict(zip(db_context.knowledge_graph.entities, db_context.entity_tokens))
        entity_tokens = []
        for e in db_context.knowledge_graph.entities:
            if e.startswith('column:'):
                table_name, column_name = e.split(':')[-2:]
                table_tokens = entity_token_map['table:'+table_name]
                column_tokens = entity_token_map[e]
                if column_name.startswith(table_name):
                    column_tokens = column_tokens[len(table_tokens):]
                entity_tokens.append(table_tokens + [Token(text='[unused30]')] + column_tokens)
            else:
                entity_tokens.append(entity_token_map[e])
        for e in entity_tokens:
            combined_tokens += [Token(text='[SEP]')] + e
        if len(combined_tokens) > 450:
            return None
        db_context.entity_tokens = entity_tokens
        fields["utterance"] = TextField(combined_tokens, self._utterance_token_indexers)

        world = SpiderWorld(db_context, query=sql)

        action_sequence, all_actions = world.get_action_sequence_and_all_actions()

        if action_sequence is None and self._keep_if_unparsable:
            # print("Parse error")
            action_sequence = []
        elif action_sequence is None:
            return None

        index_fields: List[Field] = []
        production_rule_fields: List[Field] = []

        for production_rule in all_actions:
            nonterminal, _ = production_rule.split(' -> ')
            production_rule = ' '.join(production_rule.split(' '))
            field = ProductionRuleField(production_rule,
                                        world.is_global_rule(nonterminal),
                                        nonterminal=nonterminal)
            production_rule_fields.append(field)

        valid_actions_field = ListField(production_rule_fields)
        fields["valid_actions"] = valid_actions_field

        action_map = {action.rule: i  # type: ignore
                      for i, action in enumerate(valid_actions_field.field_list)}

        for production_rule in action_sequence:
            index_fields.append(IndexField(action_map[production_rule], valid_actions_field))
        if not action_sequence:
            index_fields = [IndexField(-1, valid_actions_field)]

        action_sequence_field = ListField(index_fields)
        fields["action_sequence"] = action_sequence_field
        fields["world"] = MetadataField(world)
        fields["schema"] = table_field
        return Instance(fields)
