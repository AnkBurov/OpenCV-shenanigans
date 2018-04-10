package io.opencv.first.matrixanalysis;

import io.opencv.OpenCvBased;
import io.opencv.util.Matrices;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import static io.opencv.first.matrixanalysis.RenameShit.writeImage;
import static org.opencv.core.CvType.CV_8UC3;
import static org.opencv.imgcodecs.Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.opencv.imgproc.Imgproc.CHAIN_APPROX_NONE;
import static org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE;
import static org.opencv.imgproc.Imgproc.RETR_EXTERNAL;
import static org.opencv.imgproc.Imgproc.RETR_TREE;

public class CentralShapeMain extends OpenCvBased {

    public static void main(String[] args) throws IOException {
        File file = new ClassPathResource("images/132_029eac4e-05b7-477d-b903-996bc61e622f.jpg").getFile();

        try (Matrices matrices = new Matrices()) {
            Mat orig = matrices.fromSupplier("orig", () -> Imgcodecs.imread(file.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE));

//            Mat gauss = matrices.newMatrix("gauss");
//            Imgproc.GaussianBlur(orig, gauss, new Size(5, 5), 5);
//            writeImage(gauss, file, "_gauss.jpg");

//            Mat canny = matrices.newMatrix("canny");
//            Imgproc.Canny(orig, canny, 1, 30);
//            writeImage(canny, file, "_canny.jpg");


            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = matrices.newMatrix("hierarchy");
            Imgproc.findContours(orig, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

            contours.sort(Comparator.comparing(Imgproc::contourArea));

            Mat contoursMat = matrices.fromSupplier("contours", () -> new Mat(new Size(orig.width(), orig.height()), CV_8UC3));
            Imgproc.drawContours(contoursMat, contours, -1, new Scalar(500));
//            Imgproc.drawContours(contoursMat, Collections.singletonList(contours.get(contours.size() - 1)), -1, new Scalar(500));
            writeImage(contoursMat, file, "_contours.jpg");
        }
    }
}
