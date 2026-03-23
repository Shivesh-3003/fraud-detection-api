"""
Unit Tests for Fraud Detection ML Service

Tests the core Python ML inference components using mock models
(randomly initialised weights) so no trained .pth files are used.

"""

import numpy as np
import pytest
import torch
import joblib
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python-ml-service"))

from app.ml_models import Autoencoder, FraudClassifier
from app.inference import (
    FraudDetectionPipeline,
    AUTOENCODER_INPUT_DIM,
    CLASSIFIER_INPUT_DIM,
    FEATURE_NAMES_32,
)
from app.explainer import FraudExplainer


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def sample_transaction():
    """First row of the original Kaggle dataset."""
    return {
        "v_features": [
            -1.359807, -0.072781, 2.536347, 1.378155, -0.338321,
            0.462388, 0.239599, 0.098698, 0.363787, 0.090794,
            -0.551600, -0.617801, -0.991390, -0.311169, 1.468177,
            -0.470401, 0.207971, 0.025791, 0.403993, 0.251412,
            -0.018307, 0.277838, -0.110474, 0.066928, 0.128539,
            -0.189115, 0.133558, -0.021053,
        ],
        "amount": 149.62,
        "time": 0.0,
    }


@pytest.fixture
def mock_pipeline(tmp_path):
    """Pipeline with random weights — tests logic, not accuracy."""
    # Fit a scaler on random data so it has valid mean/std
    scaler = StandardScaler()
    scaler.fit(np.random.randn(200, AUTOENCODER_INPUT_DIM).astype(np.float32))
    joblib.dump(scaler, tmp_path / "scaler.pkl")

    # Save random model weights
    torch.save(Autoencoder(AUTOENCODER_INPUT_DIM).state_dict(), tmp_path / "autoencoder_model.pth")
    torch.save(FraudClassifier(CLASSIFIER_INPUT_DIM).state_dict(), tmp_path / "mlp_classifier.pth")

    pipeline = FraudDetectionPipeline(models_dir=str(tmp_path))
    pipeline.load_models()
    return pipeline


@pytest.fixture
def mock_explainer(mock_pipeline):
    """Explainer built on a random-weight classifier."""
    return FraudExplainer(classifier=mock_pipeline.classifier)


# ===========================================================================
# PREPROCESSING
# ===========================================================================

class TestPreprocessing:

    def test_output_shape(self, mock_pipeline, sample_transaction):
        """28 V-features + Amount_Log + Hour_sin + Hour_cos = 31."""
        features = mock_pipeline.preprocess(
            sample_transaction["v_features"],
            sample_transaction["amount"],
            sample_transaction["time"],
        )
        assert features.shape == (31,)

    def test_amount_log_transform(self, mock_pipeline):
        """Amount is stored as log1p(Amount) at index 28."""
        features = mock_pipeline.preprocess([0.0] * 28, amount=100.0, time=0.0)
        assert np.isclose(features[28], np.log1p(100.0))

    def test_cyclic_time_midnight(self, mock_pipeline):
        """At time=0 (midnight): sin=0, cos=1."""
        features = mock_pipeline.preprocess([0.0] * 28, amount=1.0, time=0.0)
        assert np.isclose(features[29], 0.0, atol=1e-6)
        assert np.isclose(features[30], 1.0, atol=1e-6)

    def test_wrong_feature_count_raises(self, mock_pipeline):
        with pytest.raises(ValueError):
            mock_pipeline.preprocess([0.0] * 27, amount=1.0, time=0.0)

    def test_cyclic_encoding_at_6am(self, mock_pipeline):
        """At 6AM: sin(π/2)=1, cos(π/2)=0."""
        features = mock_pipeline.preprocess([0.0] * 28, amount=1.0, time=6 * 3600.0)
        assert np.isclose(features[29], 1.0, atol=1e-6)
        assert np.isclose(features[30], 0.0, atol=1e-6)

    def test_cyclic_encoding_at_noon(self, mock_pipeline):
        """At 12PM: sin(π)≈0, cos(π)=-1."""
        features = mock_pipeline.preprocess([0.0] * 28, amount=1.0, time=12 * 3600.0)
        assert np.isclose(features[29], 0.0, atol=1e-6)
        assert np.isclose(features[30], -1.0, atol=1e-6)

    def test_cyclic_encoding_at_6pm(self, mock_pipeline):
        """At 6PM: sin(3π/2)=-1, cos(3π/2)≈0."""
        features = mock_pipeline.preprocess([0.0] * 28, amount=1.0, time=18 * 3600.0)
        assert np.isclose(features[29], -1.0, atol=1e-6)
        assert np.isclose(features[30], 0.0, atol=1e-6)

    def test_hour_wraps_at_24(self, mock_pipeline):
        """time=0 and time=24h must produce identical cyclic features (hour % 24 = 0)."""
        f0 = mock_pipeline.preprocess([0.0] * 28, amount=1.0, time=0.0)
        f24 = mock_pipeline.preprocess([0.0] * 28, amount=1.0, time=24 * 3600.0)
        assert np.allclose(f0[29:31], f24[29:31], atol=1e-6)

    def test_negative_amount_produces_negative_log(self, mock_pipeline):
        """Amounts in (-1, 0) produce a finite but negative Amount_Log."""
        features = mock_pipeline.preprocess([0.0] * 28, amount=-0.5, time=0.0)
        assert features[28] < 0
        assert np.isfinite(features[28])


# ===========================================================================
# MODEL ARCHITECTURE
# ===========================================================================

class TestModels:

    def test_autoencoder_reconstructs_same_shape(self):
        ae = Autoencoder(AUTOENCODER_INPUT_DIM)
        x = torch.randn(1, AUTOENCODER_INPUT_DIM)
        assert ae(x).shape == x.shape

    def test_autoencoder_error_non_negative(self):
        ae = Autoencoder(AUTOENCODER_INPUT_DIM)
        errors = ae.get_reconstruction_error(torch.randn(5, AUTOENCODER_INPUT_DIM))
        assert (errors >= 0).all()

    def test_classifier_proba_range(self):
        clf = FraudClassifier(CLASSIFIER_INPUT_DIM)
        probs = clf.predict_proba(torch.randn(10, CLASSIFIER_INPUT_DIM))
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_autoencoder_bottleneck_is_8_dim(self):
        """Encoder must compress 31 features down to an 8-dimensional bottleneck."""
        ae = Autoencoder(AUTOENCODER_INPUT_DIM)
        x = torch.randn(4, AUTOENCODER_INPUT_DIM)
        bottleneck = ae.encoder(x)
        assert bottleneck.shape == (4, 8)

    def test_classifier_deterministic_in_eval_mode(self):
        """In eval mode, dropout is disabled so repeated calls must return identical results."""
        clf = FraudClassifier(CLASSIFIER_INPUT_DIM)
        clf.eval()
        x = torch.randn(1, CLASSIFIER_INPUT_DIM)
        with torch.no_grad():
            p1 = clf.predict_proba(x)
            p2 = clf.predict_proba(x)
        assert torch.allclose(p1, p2)


# ===========================================================================
# FULL PIPELINE
# ===========================================================================

class TestPipeline:

    def test_predict_returns_valid_probability(self, mock_pipeline, sample_transaction):
        prob, _, _ = mock_pipeline.predict(
            sample_transaction["v_features"],
            sample_transaction["amount"],
            sample_transaction["time"],
        )
        assert 0.0 <= prob <= 1.0

    def test_classifier_input_is_32_features(self, mock_pipeline, sample_transaction):
        """31 scaled features + reconstruction error = 32."""
        _, _, clf_input = mock_pipeline.predict(
            sample_transaction["v_features"],
            sample_transaction["amount"],
            sample_transaction["time"],
        )
        assert clf_input.shape == (32,)

    def test_recon_error_is_last_feature(self, mock_pipeline, sample_transaction):
        """The 32nd value fed to the classifier should be the reconstruction error."""
        _, recon_error, clf_input = mock_pipeline.predict(
            sample_transaction["v_features"],
            sample_transaction["amount"],
            sample_transaction["time"],
        )
        assert np.isclose(clf_input[31], recon_error)

    def test_missing_models_raises(self, tmp_path):
        pipeline = FraudDetectionPipeline(models_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            pipeline.load_models()

    def test_determinism(self, mock_pipeline, sample_transaction):
        """Same input must produce identical outputs on repeated calls."""
        p1, e1, _ = mock_pipeline.predict(
            sample_transaction["v_features"],
            sample_transaction["amount"],
            sample_transaction["time"],
        )
        p2, e2, _ = mock_pipeline.predict(
            sample_transaction["v_features"],
            sample_transaction["amount"],
            sample_transaction["time"],
        )
        assert p1 == p2
        assert e1 == e2

    def test_batch_prediction(self, mock_pipeline, sample_transaction):
        """predict_batch returns one result per transaction with valid probability and error."""
        results = mock_pipeline.predict_batch([sample_transaction, sample_transaction])
        assert len(results) == 2
        for prob, error in results:
            assert 0.0 <= prob <= 1.0
            assert error >= 0.0

    def test_is_ready_before_and_after_loading(self, tmp_path):
        """is_ready() must be False before load_models() and True after."""
        pipeline = FraudDetectionPipeline(models_dir=str(tmp_path))
        assert not pipeline.is_ready()

        scaler = StandardScaler()
        scaler.fit(np.random.randn(200, AUTOENCODER_INPUT_DIM).astype(np.float32))
        joblib.dump(scaler, tmp_path / "scaler.pkl")
        torch.save(Autoencoder(AUTOENCODER_INPUT_DIM).state_dict(), tmp_path / "autoencoder_model.pth")
        torch.save(FraudClassifier(CLASSIFIER_INPUT_DIM).state_dict(), tmp_path / "mlp_classifier.pth")

        pipeline.load_models()
        assert pipeline.is_ready()


# ===========================================================================
# SHAP EXPLAINER
# ===========================================================================

class TestSHAP:

    def test_explain_returns_32_feature_keys(self, mock_explainer):
        """explain() must return a dict keyed by all 32 feature names."""
        clf_input = np.zeros(32, dtype=np.float32)
        shap_dict, _ = mock_explainer.explain(clf_input, nsamples=10)
        assert set(shap_dict.keys()) == set(FEATURE_NAMES_32)

    def test_shap_values_are_finite(self, mock_explainer):
        """No SHAP value or base value may be NaN or Inf."""
        clf_input = np.zeros(32, dtype=np.float32)
        shap_dict, base_value = mock_explainer.explain(clf_input, nsamples=10)
        for name, val in shap_dict.items():
            assert np.isfinite(val), f"SHAP value for {name} is not finite"
        assert np.isfinite(base_value)

    def test_base_value_in_unit_interval(self, mock_explainer):
        """base_value is the expected prediction on the background, so it must be in [0, 1]."""
        clf_input = np.zeros(32, dtype=np.float32)
        _, base_value = mock_explainer.explain(clf_input, nsamples=10)
        assert 0.0 <= base_value <= 1.0

    def test_shap_additivity(self, mock_explainer):
        """sum(SHAP values) + base_value ≈ model prediction (SHAP additivity property)."""
        clf_input = np.random.randn(32).astype(np.float32)
        shap_dict, base_value = mock_explainer.explain(clf_input, nsamples=50)

        shap_sum = sum(shap_dict.values())

        with torch.no_grad():
            tensor = torch.tensor(clf_input, dtype=torch.float32).unsqueeze(0)
            prediction = mock_explainer.classifier.predict_proba(tensor).item()

        assert abs(shap_sum + base_value - prediction) < 0.05, (
            f"SHAP additivity violated: sum={shap_sum:.4f}, base={base_value:.4f}, "
            f"prediction={prediction:.4f}, diff={abs(shap_sum + base_value - prediction):.4f}"
        )
