/*
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain slide_in_right copy of
 * the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

package hr.fer.zemris.diplomski.audio;

import android.content.SharedPreferences;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import hr.fer.zemris.diplomski.Util;

public class AudioRecorder {
    public static final int SAMPLE_RATE = 16000;

    private boolean shouldContinue;
    private AudioListener listener;
    private Thread thread;


    public AudioRecorder(AudioListener listener) {
        this.listener = listener;
    }

    public boolean recording() {
        return thread != null;
    }

    public void startRecording() {
        if (thread != null) return;

        shouldContinue = true;
        thread = new Thread(new Runnable() {
            @Override
            public void run() {
                record();
            }
        });
        thread.start();
    }

    public void stopRecording() {
        if (thread == null) return;

        shouldContinue = false;
        thread = null;
    }

    private void record() {
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);

        float inferenceLength = Util.getInferenceLength();

        int bufferSize = (int) (SAMPLE_RATE * inferenceLength);
        float[] audioBuffer = new float[bufferSize];

        AudioRecord record = new AudioRecord(MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_FLOAT,
                bufferSize);

        if (record.getState() != AudioRecord.STATE_INITIALIZED) {
            return;
        }
        record.startRecording();

        while (shouldContinue) {
            record.read(audioBuffer, 0, bufferSize, AudioRecord.READ_BLOCKING);
            listener.onAudioDataReceived(audioBuffer);
        }

        record.stop();
        record.release();
    }
}
