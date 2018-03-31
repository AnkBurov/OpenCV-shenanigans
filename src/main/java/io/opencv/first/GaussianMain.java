package io.opencv.first;

import io.opencv.OpenCvBased;
import io.opencv.util.Matrices;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.IOException;

public class GaussianMain extends OpenCvBased {
    public static void main(String[] args) throws IOException {
        File file = new ClassPathResource("images/cat.jpg").getFile();

        try (Matrices matrices = new Matrices()) {

            Mat orig = matrices.fromSupplier("orig", () -> Imgcodecs.imread(file.getAbsolutePath()/*, CV_LOAD_IMAGE_GRAYSCALE*/));
            Mat dst = matrices.newMatrix("new");
            Imgproc.GaussianBlur(orig, dst, new Size(19, 19), 5);

            Imgcodecs.imwrite("cat_gaussian.jpg", dst);
        }
    }
}
