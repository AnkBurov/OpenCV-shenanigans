package io.opencv.first.matrixanalysis.histogram;

import io.opencv.OpenCvBased;
import io.opencv.util.Matrices;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.IOException;

import static org.opencv.imgcodecs.Imgcodecs.IMREAD_GRAYSCALE;

public class Luminance extends OpenCvBased {
    public static void main(String[] args) throws IOException {
        File file = new ClassPathResource("images/spacex.jpg").getFile();

        try (Matrices matrices = new Matrices()) {
            Mat orig = matrices.fromSupplier("orig", () -> Imgcodecs.imread(file.getAbsolutePath(), IMREAD_GRAYSCALE));

            int totalIntensity = 0;
            for (int row = 0; row < orig.rows(); row++) {
                for (int column = 0; column < orig.cols(); column++) {
                    orig.get(row, column);
                    totalIntensity += orig.get(row, column)[0];
                }
            }

            double avgLum = totalIntensity / (orig.rows() * orig.cols());

            System.out.println(avgLum);
        }
    }
}
