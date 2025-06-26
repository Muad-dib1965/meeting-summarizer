import streamlit as st
import whisper
import openai
import tempfile
import os
import textwrap
from pydub import AudioSegment
from pyannote.audio import Pipeline

# --- Streamlit UI ---
st.set_page_config(page_title="Meeting Summarizer for Fixed Income", layout="centered")
st.title("📊 Fixed Income Meeting Summarizer")

st.markdown("""
Upload an audio file of your meeting (MP3 or WAV), and this app will:
1. Perform **speaker diarization**
2. Transcribe each speaker's part using **Whisper**
3. Generate a **summary** using GPT-4, focused on fixed income analysis
""")

# --- Inputs ---
audio_file = st.file_uploader("📤 Upload your meeting audio (MP3 or WAV)", type=["mp3", "wav"])
whisper_model = st.selectbox("🎯 Whisper model size", ["tiny", "base", "small", "medium", "large"], index=1)
hf_token = st.text_input("🔐 HuggingFace Token (for diarization)", type="password")
gpt_key = st.text_input("🧠 OpenAI API Key", type="password")

if st.button("▶️ Transcribe and Summarize") and audio_file and hf_token and gpt_key:
    with st.spinner("Processing audio and generating transcript..."):
        # Save uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + audio_file.name.split(".")[-1]) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        # Convert to WAV mono 16kHz
        audio = AudioSegment.from_file(tmp_path)
        wav_path = tmp_path.replace(".mp3", ".wav") if tmp_path.endswith(".mp3") else tmp_path
        audio.set_channels(1).set_frame_rate(16000).export(wav_path, format="wav")

        # Run Pyannote diarization
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
        diarization = pipeline(wav_path)

        # Load Whisper
        model = whisper.load_model(whisper_model)
        audio = AudioSegment.from_wav(wav_path)

        # Transcribe segments
        transcript = ""
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_ms = int(turn.start * 1000)
            end_ms = int(turn.end * 1000)
            segment = audio[start_ms:end_ms]

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                segment.export(temp_audio.name, format="wav")
                result = model.transcribe(temp_audio.name, fp16=False)
                os.remove(temp_audio.name)

            transcript += f"{speaker}: {result['text'].strip()}\n"

        st.subheader("🗣 Full Transcript")
        st.text_area("Transcript", transcript, height=300)

        # Chunk and summarize
        def chunk_text(text, max_chars=3000):
            return textwrap.wrap(text, max_chars, break_long_words=False)

        openai.api_key = gpt_key
        chunks = chunk_text(transcript)
        summaries = []

        for i, chunk in enumerate(chunks):
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You're an AI analyst summarizing a meeting for a fixed income investment team."},
                    {"role": "user", "content": f"Summarize the following transcript:\n\n{chunk}"}
                ],
                temperature=0.3,
            )
            summaries.append(response['choices'][0]['message']['content'])

        full_summary = "\n\n".join(summaries)
        st.subheader("📄 Summary")
        st.text_area("Summary", full_summary, height=300)

        # Optional export
        st.download_button("💾 Download Transcript", transcript, file_name="transcript.txt")
        st.download_button("💾 Download Summary", full_summary, file_name="summary.txt")
