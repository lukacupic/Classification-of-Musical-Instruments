package hr.fer.zemris.diplomski.audio;

import android.content.res.AssetManager;
import hr.fer.zemris.diplomski.Util;
import hr.fer.zemris.diplomski.mfcc.MFCC;
import org.tensorflow.lite.Interpreter;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class AudioPredictor {

    private List<String> labels;
    private Interpreter tfLite;
    private Config config;
    private MFCC mfcc;

    private int[] shape = new int[]{13, 11};
    private float min;
    private float max;

    public AudioPredictor(Interpreter tfLite, List<String> labels, AssetManager assets) {
        this.tfLite = tfLite;
        this.labels = labels;
        this.config = new Config("config.ini", assets);

        this.min = config.getFloat("min");
        this.max = config.getFloat("max");

        this.mfcc = new MFCC(config.getInt("nmfcc"), 0, config.getInt("nfft"),
                config.getInt("hop_length"), 128, AudioRecorder.SAMPLE_RATE);
    }

    public float[] predict(float[] data) {
        final float[] prediction = new float[labels.size()];

        int step = AudioRecorder.SAMPLE_RATE / 10;
        int c = 0;
        for (int i = 0; i < data.length - step; i += step) {
            float[] sample = new float[step];
            for (int j = i, k = 0; j < i + step; j++, k++) {
                sample[k] = data[j];
            }

            float[][] mfccs = mfcc.dctMfcc(sample);
            normalize(mfccs);

            float[] p = run(mfccs);
            for (int j = 0; j < p.length; j++) {
                prediction[j] += p[j];
            }
            c++;
        }

        for (int i = 0; i < prediction.length; i++) {
            prediction[i] /= c;
        }
        return prediction;
    }

    private void normalize(float[][] X) {
        float diff = max - min;

        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < X[i].length; j++) {
                X[i][j] = (X[i][j] - min) / diff;
            }
        }
    }

    private float[] run(float[][] x) {
        float[][] outputScores = new float[1][labels.size()];

        tfLite.run(reshape(x), outputScores);
        return outputScores[0];
    }

    private float[][][][] reshape(float[][] x) {
        float[][][][] X = new float[1][shape[0]][shape[1]][1];
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                X[0][i][j][0] = x[i][j];
            }
        }
        return X;
    }

    private static class Config {
        private Map<String, String> map;

        public Config(String filename, AssetManager assets) {
            loadConfig(filename, assets);
        }

        private void loadConfig(String filename, AssetManager assets) {
            Map<String, String> config = new HashMap<>();

            List<String> lines = Util.loadTextFile(filename, assets);
            for (String line : lines) {
                if (!line.contains("=")) continue;

                String[] parts = line.split("=");
                config.put(parts[0].trim(), parts[1].trim());
            }
            this.map = config;
        }

        public float getFloat(String name) {
            String value = map.get(name);
            return Float.parseFloat(value);
        }

        public int getInt(String name) {
            String value = map.get(name);
            return Integer.parseInt(value);
        }
    }
}
