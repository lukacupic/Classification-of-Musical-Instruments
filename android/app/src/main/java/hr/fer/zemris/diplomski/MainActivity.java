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

package hr.fer.zemris.diplomski;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.ColorStateList;
import android.graphics.Color;
import android.os.Bundle;
import android.widget.Button;
import androidx.annotation.NonNull;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.snackbar.Snackbar;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.appcompat.app.AppCompatActivity;
import android.view.View;
import android.widget.TextView;
import com.github.mikephil.charting.charts.BarChart;
import com.github.mikephil.charting.components.XAxis;
import com.github.mikephil.charting.components.YAxis;
import com.github.mikephil.charting.data.BarData;
import com.github.mikephil.charting.data.BarDataSet;
import com.github.mikephil.charting.data.BarEntry;
import com.github.mikephil.charting.data.Entry;
import hr.fer.zemris.diplomski.audio.AudioListener;
import hr.fer.zemris.diplomski.audio.AudioPredictor;
import hr.fer.zemris.diplomski.audio.AudioRecorder;
import org.tensorflow.lite.Interpreter;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Scanner;

public class MainActivity extends AppCompatActivity {

    private static final String MODEL_FILENAME = "model.tflite";
    private static final String LABEL_FILENAME = "labels.txt";
    private static final int REQUEST_RECORD_AUDIO = 13;

    private List<String> labels = new ArrayList<>();

    private AudioRecorder recorder;
    private AudioPredictor predictor;

    private BarChart chart;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        recorder = new AudioRecorder(new AudioListener() {
            @Override
            public void onAudioDataReceived(float[] data) {
                float[] prediction = predictor.predict(data);
                updateText(prediction);
                updateGraph(prediction);
            }
        });

        Button settingsButton = findViewById(R.id.settings_button_id);
        settingsButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // stop recording
                recorder.stopRecording();
                updateFabColor(R.color.colorPrimary);
                updateRipple(false);
                hideTextView();

                // open settings
                Intent intent = new Intent(MainActivity.this, SettingsActivity.class);
                overridePendingTransition(R.anim.slide_in_up, R.anim.slide_in_down);
                startActivity(intent);
            }
        });

        FloatingActionButton fab = findViewById(R.id.fab);
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (!recorder.recording()) {
                    startAudioRecordingSafe();
                    updateFabColor(R.color.colorAccent);
                    updateRipple(true);
                    showTextView();

                } else {
                    recorder.stopRecording();
                    updateFabColor(R.color.colorPrimary);
                    updateRipple(false);
                    hideTextView();
                }
            }
        });

        Util.setContext(getBaseContext());

        labels = Util.loadTextFile(LABEL_FILENAME, getAssets());
        Interpreter interpreter = Util.loadModel(MODEL_FILENAME, getAssets());
        predictor = new AudioPredictor(interpreter, labels, getAssets());

        createGraph();
    }

    private void createGraph() {
        chart = findViewById(R.id.chart);
        chart.setVisibility(View.INVISIBLE);

        List<BarEntry> barEntries = new ArrayList<>();
        for (int i = 0; i < this.labels.size(); i++) {
            barEntries.add(new BarEntry(0.1f, i));
        }
        BarDataSet barDataset = new BarDataSet(barEntries, "");

        BarData data = new BarData(this.labels, barDataset);
        chart.setData(data);

        YAxis left = chart.getAxisLeft();
        left.setDrawLabels(false); // no axis labels
        left.setDrawAxisLine(false); // no axis line
        left.setDrawGridLines(false); // no grid lines
        left.setDrawZeroLine(true); // draw slide_in_right zero line

        YAxis right = chart.getAxisRight();
        right.setDrawLabels(false); // no axis labels
        right.setDrawAxisLine(false); // no axis line
        right.setDrawGridLines(false); // no grid lines
        right.setDrawZeroLine(true); // draw slide_in_right zero line

        XAxis xAxis = chart.getXAxis();
        xAxis.setDrawLabels(false); // no axis labels
        xAxis.setDrawAxisLine(false); // no axis line
        xAxis.setDrawGridLines(false); // no grid lines

        chart.setDrawBorders(false);
        chart.setVisibleXRange(1f, this.labels.size());
        chart.getXAxis().setTextColor(Color.TRANSPARENT);
        chart.getXAxis().setGridColor(Color.TRANSPARENT);
    }

    private List<Entry> floatsToEntry(float[] floats) {
        List<Entry> entries = new ArrayList<>();

        for (int i = 0; i < floats.length; i++) {
            float f = floats[i];
            entries.add(new Entry(f, i));
        }

        return entries;
    }

    private void updateText(float[] prediction) {
        float max = Float.NEGATIVE_INFINITY;
        int index = 0;
        for (int i = 0; i < prediction.length; i++) {
            if (prediction[i] > max) {
                max = prediction[i];
                index = i;
            }
        }

        String instrument = String.format(Locale.US, "%s (%.4f%%)", labels.get(index), 100 * prediction[index]);
        updateTextView(instrument);
    }

    private void updateGraph(final float[] prediction) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if (Double.isNaN(prediction[0])) return;

                List<BarEntry> barEntries = new ArrayList<>();
                List<Entry> f2e = floatsToEntry(prediction);
                for (Entry entry : f2e) {
                    barEntries.add(new BarEntry(entry.getVal(), entry.getXIndex()));
                }
                BarDataSet barDataset = new BarDataSet(barEntries, "X");

                BarData data = new BarData(labels, barDataset);
                chart.setData(data);
                chart.notifyDataSetChanged();
                chart.invalidate();
            }
        });
    }

    private void updateRipple(final boolean bool) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                PulsatingButton button = findViewById(R.id.fab);
                button.animateButton(bool);
            }
        });
    }

    public void updateTextView(final String content) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                TextView textView = findViewById(R.id.instrumentView);
                textView.setText(content);
            }
        });
    }

    public void hideTextView() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                TextView textView = findViewById(R.id.instrumentView);
                textView.setVisibility(View.INVISIBLE);
            }
        });
    }

    public void showTextView() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                TextView textView = findViewById(R.id.instrumentView);
                textView.setVisibility(View.VISIBLE);
            }
        });
    }

    @Override
    protected void onStop() {
        super.onStop();
        recorder.stopRecording();
    }

    private void startAudioRecordingSafe() {
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            recorder.startRecording();
        } else {
            requestMicrophonePermission();
        }
    }

    private void updateFabColor(final int color) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                FloatingActionButton fab = findViewById(R.id.fab);
                fab.setBackgroundTintList(ColorStateList.valueOf(ContextCompat.getColor(MainActivity.this, color)));
            }
        });
    }

    private void requestMicrophonePermission() {
        if (ActivityCompat.shouldShowRequestPermissionRationale(this, android.Manifest.permission.RECORD_AUDIO)) {
            Snackbar.make(null, "Microphone access is required in order to record audio",
                    Snackbar.LENGTH_INDEFINITE).setAction("OK", new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    ActivityCompat.requestPermissions(MainActivity.this, new String[]{
                            android.Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
                }
            }).show();
        } else {
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{
                    android.Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == REQUEST_RECORD_AUDIO && grantResults.length > 0 &&
                grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            recorder.stopRecording();
        }
    }
}
