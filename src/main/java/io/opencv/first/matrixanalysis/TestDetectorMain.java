package io.opencv.first.matrixanalysis;

import io.opencv.OpenCvBased;
import io.opencv.util.Matrices;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class TestDetectorMain extends OpenCvBased {

    public static void main(String[] args) throws Exception {
        File file = new ClassPathResource("images/vin.jpeg").getFile();
//        File file = new ClassPathResource("images/vin2.jpeg").getFile();

        try (Matrices matrices = new Matrices()) {
            Mat orig_pic = matrices.fromSupplier("orig_pic", () -> Imgcodecs.imread(file.getAbsolutePath()));

            List<Rect> letterBBoxes1 = detectLetters(orig_pic);

            for (int i = 0; i < letterBBoxes1.size(); i++)
                Imgproc.rectangle(orig_pic, letterBBoxes1.get(i).br(), letterBBoxes1.get(i).tl(), new Scalar(0, 255, 0), 3, 8, 0);
            Imgcodecs.imwrite("abc1.png", orig_pic);
        }
    }

    public static List<Rect> detectLetters(Mat img) {
        List<Rect> boundRect = new ArrayList<>();

        Mat img_gray = new Mat(), img_sobel = new Mat(), img_threshold = new Mat(), element = new Mat();

        Imgproc.cvtColor(img, img_gray, Imgproc.COLOR_RGB2GRAY);

        Imgproc.GaussianBlur(img_gray, img_gray, new Size(3, 3), 30);

        Imgproc.Sobel(img_gray, img_sobel, CvType.CV_8U, 1, 0, 3, 1, 0, Core.BORDER_DEFAULT);
        //at src, Mat dst, double thresh, double maxval, int type
        Imgproc.threshold(img_sobel, img_threshold, 0, 255, 8);
        element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(15, 5));
        Imgproc.morphologyEx(img_threshold, img_threshold, Imgproc.MORPH_CLOSE, element);
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(img_threshold, contours, hierarchy, 0, 1);

        List<MatOfPoint> contours_poly = new ArrayList<MatOfPoint>(contours.size());

        for (int i = 0; i < contours.size(); i++) {

            MatOfPoint2f mMOP2f1 = new MatOfPoint2f();
            MatOfPoint2f mMOP2f2 = new MatOfPoint2f();

            contours.get(i).convertTo(mMOP2f1, CvType.CV_32FC2);
            Imgproc.approxPolyDP(mMOP2f1, mMOP2f2, 2, true);
            mMOP2f2.convertTo(contours.get(i), CvType.CV_32S);

            Rect appRect = Imgproc.boundingRect(contours.get(i));
            if (appRect.width > appRect.height) {
                boundRect.add(appRect);
            }
        }

        return boundRect;
    }
}
