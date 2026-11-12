import unittest
import json
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, model, clean_text, get_contributing_keywords

class FakeNewsDetectorTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_model_loaded(self):
        """Verify the model pipeline is loaded and ready."""
        self.assertIsNotNone(model, "Model should be loaded from model.pkl")

    def test_clean_text_utility(self):
        """Verify text preprocessing behaves properly."""
        raw = "   BREAKING: Check https://example.com for <b>urgent</b> news! \n\n  "
        cleaned = clean_text(raw)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("<b>", cleaned)
        self.assertEqual(cleaned, "breaking: check for urgent news!")

    def test_home_page_status(self):
        """Check homepage renders HTTP 200."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'AI Fake News Detector', response.data)

    def test_health_endpoint(self):
        """Check health status returns valid JSON with healthy state."""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get('status'), 'healthy')
        self.assertTrue(data.get('model_loaded'))

    def test_api_info_endpoint(self):
        """Check model info endpoint returns metrics."""
        response = self.app.get('/api/info')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get('status'), 'success')
        self.assertIn('metrics', data)

    def test_predict_real_news(self):
        """Check prediction on genuine science news."""
        payload = {
            'text': "NASA's James Webb Space Telescope discovers ancient distant galaxy cluster formed after Big Bang."
        }
        response = self.app.post('/predict',
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get('status'), 'success')
        self.assertEqual(data.get('prediction'), 'REAL')
        self.assertIn('probabilities', data)
        self.assertIn('REAL', data['probabilities'])

    def test_predict_fake_news(self):
        """Check prediction on fabricated conspiracy claim."""
        payload = {
            'text': "Secret subterranean reptilian aliens control world government through microwave frequency mind control."
        }
        response = self.app.post('/predict',
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get('status'), 'success')
        self.assertEqual(data.get('prediction'), 'FAKE')
        self.assertIn('probabilities', data)
        self.assertIn('FAKE', data['probabilities'])

    def test_predict_empty_text_validation(self):
        """Verify API rejects empty text gracefully with 400 Bad Request."""
        payload = {'text': "   "}
        response = self.app.post('/predict',
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_predict_batch(self):
        """Verify batch prediction endpoint operates on multiple inputs."""
        payload = {
            'texts': [
                "European Space Agency successfully launches satellite into lunar orbit.",
                "Drinking bleach cures viral infections overnight according to secret leak."
            ]
        }
        response = self.app.post('/predict_batch',
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get('status'), 'success')
        self.assertEqual(len(data.get('results', [])), 2)
        self.assertEqual(data['results'][0]['prediction'], 'REAL')
        self.assertEqual(data['results'][1]['prediction'], 'FAKE')

    def test_keywords_extraction(self):
        """Verify top contributing keywords are returned."""
        text = "Secret alien base found on Mars using telepathic androids."
        keywords = get_contributing_keywords(text, top_n=3)
        self.assertIsInstance(keywords, list)
        if len(keywords) > 0:
            self.assertIn('term', keywords[0])
            self.assertIn('favors', keywords[0])

if __name__ == '__main__':
    unittest.main()
