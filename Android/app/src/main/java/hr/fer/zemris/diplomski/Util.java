package hr.fer.zemris.diplomski;

import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.preference.PreferenceManager;
import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class Util {

    private static Context context;

    public static void setContext(Context context) {
        Util.context = context;
    }

    private static SharedPreferences getPreferences() {
        return PreferenceManager.getDefaultSharedPreferences(context);
    }

    public static float getInferenceLength() {
        String inferenceLength = getPreferences().getString("pref_inference_length", "1");
        return Float.parseFloat(inferenceLength);
    }

    public static List<String> loadTextFile(String filename, AssetManager assets) {
        List<String> lines = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new InputStreamReader(assets.open(filename)))) {
            String line;
            while ((line = br.readLine()) != null) {
                lines.add(line);
            }
        } catch (IOException e) {
            throw new RuntimeException("Error loading the" + filename + " file.");
        }

        return lines;
    }

    public static Interpreter loadModel(String filename, AssetManager assets) {
        Interpreter tflite;
        try {
            ByteBuffer bb = loadModelFile(assets, filename);
            tflite = new Interpreter(bb);
        } catch (Exception e) {
            throw new RuntimeException("Error loading the model file.");
        }

        tflite.resizeInput(0, new int[]{1, 20, 4, 1});
        return tflite;
    }

    private static ByteBuffer loadModelFile(AssetManager assets, String modelFilename) throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}
