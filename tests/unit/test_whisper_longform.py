#!/usr/bin/env python3
"""Tests del soporte long-form (> 30 s) en la transcripción Whisper.

Verifican desde whisper_service._transcribe_sync (sin GPU):

- Audio > 30 s: nunca se trunca, se pasa attention_mask y se fuerza
  return_timestamps=True (transformers lo exige para long-form).
- Modo "off" en audio largo: los timestamps se usan internamente pero se
  eliminan del texto final (skip_special_tokens).
- Modos segment/word: se usan los segmentos devueltos por generate()
  (return_segments=True) y texto/words derivados.
- Audio corto: se mantiene la llamada clásica (return_timestamps=False).
"""

import numpy as np
import torch

from services import whisper_service as ws

_TIMESTAMP_BEGIN = 50363


def _ts(seconds: float) -> int:
    """Id de token de tiempo para una marca en segundos."""
    return _TIMESTAMP_BEGIN + int(round(seconds / ws._TIME_PRECISION))


class FakeTokenizer:
    """Tokenizer mínimo con timestamp_ids() y decode por mapeo de ids."""

    def __init__(self, pieces: dict):
        self._pieces = pieces

    def timestamp_ids(self):
        return list(range(_TIMESTAMP_BEGIN, _TIMESTAMP_BEGIN + 1500))

    @property
    def timestamp_begin(self):
        raise AttributeError("timestamp_begin ya no existe en transformers 4.57")

    def decode(self, token_ids, skip_special_tokens=True):
        out = []
        for tid in token_ids:
            piece = self._pieces.get(int(tid))
            if piece is not None:
                out.append(piece)
        return "".join(out)


class FakeProcessor:
    """Simula WhisperProcessor: features + attention_mask + decode."""

    def __init__(self, tokenizer: FakeTokenizer, duration: float):
        self.tokenizer = tokenizer
        n = max(int(round(duration * 100)), 1)
        self._input_features = torch.zeros(1, 128, n)
        self._attention_mask = torch.ones(1, n, dtype=torch.long)

    def __call__(self, wav, sampling_rate, return_tensors, truncation,
                 padding, return_attention_mask):
        return SimpleNamespaceFE(
            self._input_features, self._attention_mask
        )

    def batch_decode(self, sequences, skip_special_tokens=True):
        seqs = sequences.tolist() if hasattr(sequences, "tolist") else sequences
        return [
            self.tokenizer.decode(s, skip_special_tokens=skip_special_tokens)
            for s in seqs
        ]

    def get_decoder_prompt_ids(self, language, task):
        return [("decoder_prompt_ids", True)]


class SimpleNamespaceFE:
    """Resultado del procesador con atributos y .get (como BatchFeature)."""

    def __init__(self, input_features, attention_mask):
        self.input_features = input_features
        self.attention_mask = attention_mask

    def get(self, key, default=None):
        return getattr(self, key, default)


class FakeAudioService:
    """devuelve un wav silencioso de la duración pedida."""

    def __init__(self, duration: float):
        self._duration = duration

    def load(self, audio, target_sr=16000):
        return np.zeros(int(target_sr * self._duration), dtype=np.float32), target_sr


def _pieces(*texts) -> dict:
    return {100 + i: t for i, t in enumerate(texts)}


def _make_svc(duration: float, pieces: dict, output):
    """WhisperService aislado con model/processor fakes y output canjeado."""
    svc = ws.WhisperService(audio_service=FakeAudioService(duration))
    model = FakeModel(output)
    svc._model = model
    svc._processor = FakeProcessor(FakeTokenizer(pieces), duration)
    svc._model_name = "whisper-large-v3"
    svc._ensure_loaded = lambda: None
    return svc, model


class FakeModel:
    device = "cpu"
    dtype = torch.float32

    def __init__(self, output):
        self._output = output
        self.calls = []

    def generate(self, input_features, **kwargs):
        self.calls.append(kwargs)
        return self._output


def _flat_ids(*groups):
    ids = []
    for group in groups:
        ids += list(group)
    return torch.tensor([ids], dtype=torch.long)


# -- Audio largo (> 30 s) ---------------------------------------------------


def test_long_audio_modo_off_fuerza_timestamps_y_los_quita_del_texto():
    """78 s + modo off: long-form con return_timestamps=True (necesario para
    el algoritmo secuencial) pero el texto final sale sin marcas de tiempo."""
    pieces = _pieces("Hola", " ", "mundo", " ", "Adiós")
    seq = _flat_ids(
        [_ts(0.0), 100, 101, 102, _ts(30.0)],
        [_ts(30.0), 104],
    )
    svc, model = _make_svc(78.0, pieces, seq)

    result = svc._transcribe_sync(object(), "es", "transcribe", timestamps="off")

    kwargs = model.calls[0]
    assert kwargs["return_timestamps"] is True
    assert kwargs["attention_mask"] is not None
    assert "return_segments" not in kwargs
    assert result["text"] == "Hola mundoAdiós"
    assert "<|" not in result["text"]
    assert "segments" not in result
    assert result["duration_seconds"] == 78.0
    assert result["timestamps"] == "off"


def test_long_audio_fuerza_idioma_y_mask_en_generate():
    """Long-form con idioma forzado: forced_decoder_ids y attention_mask."""
    pieces = _pieces("Hola", " ", "mundo")
    seq = _flat_ids([_ts(0.0), 100, 101, 102, _ts(30.0)])
    svc, model = _make_svc(78.0, pieces, seq)

    result = svc._transcribe_sync(object(), "es", "transcribe", timestamps="off")

    kwargs = model.calls[0]
    assert kwargs["forced_decoder_ids"] is not None
    assert kwargs["return_timestamps"] is True
    assert kwargs["attention_mask"] is not None
    assert result["language"] == "es"


def test_long_audio_segmentos_desde_generate():
    """Segmentos long-form devueltos por generate() (return_segments=True)."""
    pieces = _pieces("Hola", " ", "mundo", " ", "Adiós")
    matched = {0.0: 0.0, 30.0: 30.0, 60.0: 60.0}

    def seg_obj(ts):
        s = matched[float(ts)]
        return torch.tensor([s], dtype=torch.float64)

    segments = [
        [
            {
                "start": seg_obj(0.0),
                "end": seg_obj(30.0),
                "tokens": torch.tensor([_ts(0.0), 100, 101, 102, _ts(30.0)]),
            },
            {
                "start": seg_obj(30.0),
                "end": seg_obj(60.0),
                "tokens": torch.tensor([_ts(30.0), 104]),
            },
        ]
    ]
    seq = _flat_ids(
        [_ts(0.0), 100, 101, 102, _ts(30.0)],
        [_ts(30.0), 104],
    )
    output = {"sequences": seq, "segments": segments}
    svc, model = _make_svc(78.0, pieces, output)

    result = svc._transcribe_sync(object(), "es", "transcribe", timestamps="segment")

    kwargs = model.calls[0]
    assert kwargs["return_timestamps"] is True
    assert kwargs["return_segments"] is True
    assert result["segments"] == [
        {"start": 0.0, "end": 30.0, "text": "Hola mundo"},
        {"start": 30.0, "end": 60.0, "text": "Adiós"},
    ]
    assert result["text"] == "Hola mundoAdiós"


def test_long_audio_segmentos_fallback_manual():
    """Si generate() no devuelve "segments", se reconstruyen del flujo."""
    pieces = _pieces("Hola", " ", "mundo", " ", "Adiós")
    seq = _flat_ids(
        [_ts(0.0), 100, 101, 102, _ts(30.0)],
        [_ts(30.0), 104],
    )
    # Salida tipo tensor (sin return_segments): el servicio debe caer en la
    # extracción manual.
    svc, model = _make_svc(78.0, pieces, seq)

    result = svc._transcribe_sync(object(), "es", "transcribe", timestamps="segment")

    assert result["segments"] == [
        {"start": 0.0, "end": 30.0, "text": "Hola mundo"},
        {"start": 30.0, "end": 30.0, "text": "Adiós"},
    ]


def test_long_audio_words_desde_segmentos():
    pieces = _pieces("Hola", " ", "mundo", " ", "Adiós")
    segments = [
        [
            {
                "start": 0.0,
                "end": 10.0,
                "tokens": torch.tensor([_ts(0.0), 100, 101, 102, _ts(10.0)]),
            },
            {
                "start": 10.0,
                "end": 20.0,
                "tokens": torch.tensor([_ts(10.0), 104, _ts(20.0)]),
            },
        ]
    ]
    seq = _flat_ids(
        [_ts(0.0), 100, 101, 102, _ts(10.0)],
        [_ts(10.0), 104, _ts(20.0)],
    )
    output = {"sequences": seq, "segments": segments}
    svc, model = _make_svc(78.0, pieces, output)

    result = svc._transcribe_sync(object(), "es", "transcribe", timestamps="word")

    assert result["segments"][0]["start"] == 0.0
    assert result["segments"][0]["end"] == 10.0
    assert result["segments"][0]["text"] == "Hola mundo"
    assert [w["word"] for w in result["words"]] == ["Hola", "mundo", "Adiós"]


# -- Audio corto (<= 30 s): comportamiento clásico --------------------------


def test_short_audio_modo_off_sin_timestamps():
    """Audio corto + modo off: llamada clásica (return_timestamps=False)."""
    pieces = _pieces("Hola", " ", "mundo")
    seq = torch.tensor([[100, 101, 102]], dtype=torch.long)
    svc, model = _make_svc(10.0, pieces, seq)

    result = svc._transcribe_sync(object(), "es", "transcribe", timestamps="off")

    kwargs = model.calls[0]
    assert kwargs["return_timestamps"] is False
    assert kwargs["attention_mask"] is not None
    assert "return_segments" not in kwargs
    assert result["text"] == "Hola mundo"
    assert "segments" not in result


def test_short_audio_modo_segment_con_return_segments():
    """Audio corto + segment: algoritmos secuencial y return_segments=True."""
    pieces = _pieces("Hola", " ", "mundo")
    seg_output = {
        "sequences": torch.tensor([[100, 101, 102]], dtype=torch.long),
        "segments": [[
            {
                "start": 0.0,
                "end": 0.5,
                "tokens": torch.tensor([100, 101, 102]),
            }
        ]],
    }
    svc, model = _make_svc(10.0, pieces, seg_output)

    result = svc._transcribe_sync(object(), "es", "transcribe", timestamps="segment")

    kwargs = model.calls[0]
    assert kwargs["return_timestamps"] is True
    assert kwargs["return_segments"] is True
    assert result["segments"] == [{"start": 0.0, "end": 0.5, "text": "Hola mundo"}]


def test_long_audio_es_long_form_segun_features_no_segundos():
    """El umbral es el numero de features (no la duracion redondeada)."""
    pieces = _pieces("Hola")
    seq = _flat_ids([_ts(0.0), 100, _ts(30.0)])
    svc, model = _make_svc(31.0, pieces, seq)
    # 31 s -> 3100 features (> 3000): long-form aunque "solo" 31 s.
    assert model.calls == []
    svc._transcribe_sync(object(), "es", "transcribe", timestamps="off")
    assert model.calls[0]["return_timestamps"] is True