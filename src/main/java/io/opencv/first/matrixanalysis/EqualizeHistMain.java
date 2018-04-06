package io.opencv.first.matrixanalysis;

import io.opencv.OpenCvBased;
import io.opencv.util.Matrices;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.IOException;

import static io.opencv.first.matrixanalysis.RenameShit.writeImage;
import static org.opencv.imgcodecs.Imgcodecs.IMREAD_GRAYSCALE;

public class EqualizeHistMain extends OpenCvBased {

    public static void main(String[] args) throws IOException {
        File file = new ClassPathResource("images/фото ТС 3.JPG").getFile();

        try (Matrices matrices = new Matrices()) {
            Mat orig = matrices.fromSupplier("orig", () -> Imgcodecs.imread(file.getAbsolutePath(), IMREAD_GRAYSCALE));
            writeImage(orig, file, "_orig.jpg");

            Mat equalizeHist = matrices.newMatrix("equalizeHist");
            Imgproc.equalizeHist(orig, equalizeHist);
            writeImage(equalizeHist, file, "_equalizeHist.jpg");
        }
    }
}
