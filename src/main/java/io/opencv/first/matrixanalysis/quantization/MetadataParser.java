package io.opencv.first.matrixanalysis.quantization;

import org.apache.tika.exception.TikaException;
import org.apache.tika.metadata.Metadata;
import org.apache.tika.parser.AutoDetectParser;
import org.apache.tika.parser.ParseContext;
import org.apache.tika.parser.Parser;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

public class MetadataParser {

    public Optional<String> parseModel(Path file) {
        try {
            Map<String, Object> metadata = parseMetadataMap(file);

            return Optional.ofNullable((String) metadata.get("Model"));
        } catch (IOException | SAXException | TikaException e) {
            throw new RuntimeException(e);
        }
    }

    private Map<String, Object> parseMetadataMap(Path file) throws IOException, SAXException, TikaException {
        Metadata metadata = new Metadata();
        Parser parser = new AutoDetectParser();

        parser.parse(Files.newInputStream(file), new DefaultHandler(), metadata, new ParseContext());
        final Map<String, Object> data = new HashMap<>();

        Arrays.stream(metadata.names())
              .forEach(k -> data.put(k, metadata.isMultiValued(k) ? Arrays.asList(metadata.getValues(k)) : metadata.get(k)));
        return data;
    }
}
