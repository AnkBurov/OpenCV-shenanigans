package io.opencv.first.matrixanalysis.histogram;

import io.opencv.OpenCvBased;
import io.opencv.util.Matrices;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.IOException;

import static org.opencv.imgcodecs.Imgcodecs.IMREAD_GRAYSCALE;

public class LuminanceSecond extends OpenCvBased {
    public static void main(String[] args) throws IOException {
        File file = new ClassPathResource("images/spacex.jpg").getFile();

        try (Matrices matrices = new Matrices()) {
            Mat orig = matrices.fromSupplier("orig", () -> Imgcodecs.imread(file.getAbsolutePath(), IMREAD_GRAYSCALE));

        }
    }
}
