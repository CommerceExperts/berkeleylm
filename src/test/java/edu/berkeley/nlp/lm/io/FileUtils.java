package edu.berkeley.nlp.lm.io;

import java.io.File;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.fail;

public class FileUtils
{

	/**
	 * @param testFileName
	 * @return
	 */
	public static File getFile(final String testFileName) {
		File txtFile = null;
		try {
			Path path = Paths.get(FileUtils.class.getProtectionDomain().getCodeSource().getLocation().toURI()).resolve(testFileName);
			txtFile = path.toFile();
		} catch (final URISyntaxException e) {
			fail(e.toString());
		}
		assertNotNull(txtFile);
		return txtFile;
	}

}
