package io.opencv.first.matrixanalysis;

import io.opencv.OpenCvBased;
import io.opencv.util.Matrices;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class SamePixelColorMatricsFinder extends OpenCvBased {

    private static final int MATRIX_SIZE = 30;

    public static void main(String[] args) throws IOException {
        //        File file = new ClassPathResource("images/IMG_1151_photoshoped.jpg").getFile();
//        File file = new ClassPathResource("images/14_photoshoped.jpg").getFile();
//        File file = new ClassPathResource("images/14_resaved.jpg").getFile();
//        File file = new ClassPathResource("images/14.jpg").getFile();
//        File file = new ClassPathResource("images/d7fc888s-960.jpg").getFile();
//        File file = new ClassPathResource("images/IMG_1151.jpg").getFile();
        File file = new ClassPathResource("images/spacex_photoshoped.jpg").getFile();
//        File file = new ClassPathResource("images/spacex.jpg").getFile();

        handleFile(file.getAbsolutePath(), MATRIX_SIZE);
    }

    public static void handleFile(String absolutePath, int matrixSize) {
        try (Matrices matrices = new Matrices()) {
            Mat orig = matrices.fromSupplier("orig", () -> Imgcodecs.imread(absolutePath));
            Mat dst = matrices.fromSupplier("new", orig::clone);

            List<Map.Entry<Integer, Integer>> rowColumns = new ArrayList<>();

            for (int row = 0; row < dst.rows(); row++) {
                for (int column = 0; column < dst.cols(); column++) {
                    Boolean isMatrixOfSameColor = isMatrixOfSameColor(dst, row, column, matrixSize);
                    if (isMatrixOfSameColor) {
                        System.out.println("Matrics starts on row | column " + row + "|" + column + " contains only same color");
                        rowColumns.add(new AbstractMap.SimpleEntry<>(row, column));
                    }
                }
            }

            for (Map.Entry<Integer, Integer> rowColumn : rowColumns) {
                Point startingPoing = new Point(rowColumn.getValue() - 1, rowColumn.getKey() - 1);
                Point endingPoint = new Point(rowColumn.getValue() - 1 + matrixSize, rowColumn.getKey() - 1 + matrixSize);
                Imgproc.rectangle(dst, startingPoing, endingPoint, new Scalar(500));
            }

            Imgcodecs.imwrite("analyzed.jpg", dst);
        }
    }

    private static Boolean isMatrixOfSameColor(Mat image, int row, int column, int matrixSize) {
        List<double[]> rowStartingPixels = new ArrayList<>();
        for (int matrixRow = 0; matrixRow < matrixSize; matrixRow++) {
            double[] rowStartingPixel = image.get(row + matrixRow, column);
            Boolean isSameColor = areNextColumnsInMatrixOfSameColor(image, rowStartingPixel,
                    false, row + matrixRow, column, matrixSize - 1);
            if (!Boolean.TRUE.equals(isSameColor)) {
                return false;
            }
            rowStartingPixels.add(rowStartingPixel);
        }
        //  filter values equal to other values in collection and check that all values are equal
        return rowStartingPixels.stream()
                .filter(pixel -> rowStartingPixels.stream()
                        .filter(comparingPixel -> Arrays.equals(pixel, comparingPixel))
                        .count() == rowStartingPixels.size()
                )
                .count() == rowStartingPixels.size();


        /*return rowStartingPixels.stream()
                .distinct()
                .count() == 1;*/
//        return isSameColor;
    }

    private static Boolean areNextColumnsInMatrixOfSameColor(Mat image, double[] previousPixel,
                                                            boolean isSimilar, int row, int column, int depth) {
        if (depth == 0) {
            return isSimilar;
        }

        double[] nextPixel = image.get(row, column);
        if (nextPixel == null) {
            return null;
        }
        if (Arrays.equals(previousPixel, nextPixel)) {
            return areNextColumnsInMatrixOfSameColor(image, nextPixel, true, row, column + 1, depth - 1);
        }
        return false;
    }
}
