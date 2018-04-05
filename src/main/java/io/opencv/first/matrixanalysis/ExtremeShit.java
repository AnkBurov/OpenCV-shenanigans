package io.opencv.first.matrixanalysis;

import lombok.Builder;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.opencv.core.Point;

@Getter
@Builder
public class ExtremeShit {

    private Point leftMost;
    private Point topMost;
    private Point rightMost;
    private Point bottomMost;
}
