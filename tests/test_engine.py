import tempfile
import unittest
from unittest.mock import patch
from policy_engine import Gemini, PolicyError, VectorIndex, answer, build_index, load_policies, rules_search


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.records = load_policies()

    def test_database(self):
        self.assertEqual(len(self.records), 98)
        self.assertEqual(len({p['id'] for p in self.records}), 98)

    def test_rules_preserve_exact_evidence_and_no_tokens(self):
        result = answer('How many vacation days do employees receive?', 'rules', self.records)
        self.assertEqual(result['policies'][0]['title'], 'Vacation Policy')
        self.assertIn('15 vacation days', result['answer'])
        self.assertEqual(result['llm_total_tokens'], 0)

    def test_rules_unknown_does_not_invent(self):
        result = answer('What is the pension contribution percentage?', 'rules', self.records)
        self.assertEqual(result['answer'], 'No matching policy found.')

    def test_vector_index_uses_cosine_and_validates_dimension(self):
        index = VectorIndex(self.records[:3], [[2, 0], [0, 5], [-1, 0]])
        self.assertEqual(index.search([0, 2], 1)[0]['policy'], self.records[1])
        with self.assertRaises(PolicyError):
            index.search([1, 2, 3])

    def test_full_context_and_invalid_citations(self):
        class Fake:
            model = 'test-model'
            def generate(inner, question, context):
                self.assertEqual(context, self.records)
                return {'answer': 'Example', 'policy_ids': ['P001', 'FAKE']}, {'promptTokenCount': 10, 'candidatesTokenCount': 4, 'totalTokenCount': 14}
        result = answer('Reviews?', 'full_context', self.records, Fake())
        self.assertEqual(result['invalid_policy_ids'], ['FAKE'])
        self.assertEqual(result['llm_total_tokens'], 14)
        self.assertEqual(result['review']['unsupported'], 'unreviewed')

    def test_rag_only_passes_retrieved_evidence(self):
        class Fake:
            model, embedding_model = 'test-model', 'test-embedding'
            def embed(inner, text, document=False):
                return [0, 1], None
            def generate(inner, question, context):
                self.assertEqual(context, [self.records[1]])
                return {'answer': 'Example', 'policy_ids': [self.records[1]['id']]}, {}
        index = VectorIndex(self.records[:2], [[1, 0], [0, 1]])
        result = answer('Example?', 'rag', self.records, Fake(), index, k=1)
        self.assertIsNone(result['query_embedding_tokens'])
        self.assertIsNone(result['llm_total_tokens'])

    def test_index_resume_avoids_repeat_api_calls(self):
        class Fake:
            embedding_model = 'test-embedding'
            calls = 0
            def embed(inner, text, document=False):
                inner.calls += 1
                return [1, inner.calls], None
        client = Fake()
        with tempfile.TemporaryDirectory() as directory:
            build_index(client, self.records[:2], directory)
            _, setup = build_index(client, self.records[:2], directory)
        self.assertEqual(client.calls, 2)
        self.assertEqual(setup['new_embeddings'], 0)
        self.assertIsNone(setup['index_tokens'])

    def test_api_response_parsing_preserves_usage(self):
        client = Gemini('test-key')
        fake = {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': '{"answer":"15 days","policy_ids":["P095"]}'}]}}], 'usageMetadata': {'totalTokenCount': 20}}
        with patch.object(client, 'request', return_value=fake):
            parsed, usage = client.generate('Vacation?', self.records)
        self.assertEqual(parsed['answer'], '15 days')
        self.assertEqual(usage['totalTokenCount'], 20)


if __name__ == '__main__':
    unittest.main()
