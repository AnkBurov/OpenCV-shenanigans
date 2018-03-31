package io.opencv.first;

import io.opencv.OpenCvBased;
import io.opencv.util.Matrices;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.IOException;

public class GetChecker extends OpenCvBased {
    public static void main(String[] args) throws IOException {
        File file = new ClassPathResource("images/IMG_1151_photoshoped.jpg").getFile();

        try (Matrices matrices = new Matrices()) {

            Mat orig = matrices.fromSupplier("orig", () -> Imgcodecs.imread(file.getAbsolutePath()/*, CV_LOAD_IMAGE_GRAYSCALE*/));
            Mat dst = matrices.fromSupplier("new", orig::clone);

            double[] pixel = dst.get(0, 0);
            System.out.println();
        }
    }
}
