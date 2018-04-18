package io.opencv.first.matrixanalysis.histogram;

import io.opencv.OpenCvBased;
import io.opencv.util.Matrices;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.util.Collections;

import static org.opencv.imgcodecs.Imgcodecs.IMREAD_GRAYSCALE;

public class HistogramShit extends OpenCvBased {
    public static void main(String[] args) throws Exception {
        File file = new ClassPathResource("images/spacex.jpg").getFile();

        try (Matrices matrices = new Matrices()) {
            Mat orig = matrices.fromSupplier("orig", () -> Imgcodecs.imread(file.getAbsolutePath(), IMREAD_GRAYSCALE));

            Mat histogram = matrices.newMatrix("histogram");
            Imgproc.calcHist(Collections.singletonList(orig), new MatOfInt(0), new Mat(), histogram, new MatOfInt(5), new MatOfFloat(0, 256));

            System.out.println();
        }
    }
}
