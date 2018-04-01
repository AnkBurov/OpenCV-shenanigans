package io.opencv.first.matrixanalysis;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;

public class ImagesChecker {

    private static final String[] IMG_EXTENSIONS = new String[]{".jpg", ".bmp", ".jpeg", ".png"};

    public static void main(String[] args) throws IOException {

        List<File> images = Files.walk(Paths.get("D:\\Downloads\\random photos"))
                .filter(Files::isRegularFile)
                .map(Path::toFile)
                .filter(file -> isImage(file.getName()))
                .collect(Collectors.toList());

        images.parallelStream()
                .forEach(file -> SamePixelColorMatricsFinder.handleFile(file, 20, 0.85));
    }

    private static boolean isImage(String fileName) {
        for (String imgExtension : IMG_EXTENSIONS) {
            if (fileName.endsWith(imgExtension)) {
                return true;
            }
        }
        return false;
    }
}
