package io.opencv.util;

import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Mat;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.function.Supplier;

/**
 * @author jihor (dmitriy_zhikharev@rgs.ru)
 * Created on 2017-11-29
 */
@Slf4j
public class Matrices implements AutoCloseable {
    private final Map<String, Mat> matricesMap = new HashMap<>();

    public Mat newMatrix(String key){
        return fromSupplier(key, Mat::new);
    }

    public <T extends Mat> T fromSupplier(Supplier<T> supplier){
        return fromSupplier(UUID.randomUUID().toString(), supplier);
    }

    public <T extends Mat> T fromSupplier(String key, Supplier<T> supplier){
        T mat = supplier.get();
        matricesMap.put(key, mat);
        return mat;
    }

    @Override
    public void close() {
        matricesMap.values().forEach(Matrices::quietRelease);
    }

    /**
     * Matrix objects must be released manually or a memory leak occurs, see https://github.com/opencv/opencv/issues/4961
     * <p>
     * See this nice comment from OpenCV committer:
     * "JVM doesn't know anything about of underlying native resources (for example, image buffers and their sizes). So Java heap size usage is not computer properly and GC is not called.
     * There are some working options:
     * <p>
     * Try to manually release Mat objects.
     * Invoke System.gc() every few frames.
     * There is nothing to fix in OpenCV, closing."
     *
     * @param mat - matrix to invoke release() on
     */
    private static void quietRelease(Mat mat) {
        if (mat != null) {
            try {
                mat.release();
            } catch (Throwable t) {
                log.error("Exception on releasing matrix", t);
            }
        }
    }
}
