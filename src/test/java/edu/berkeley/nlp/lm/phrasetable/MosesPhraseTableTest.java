package edu.berkeley.nlp.lm.phrasetable;

import java.util.Arrays;
import java.util.List;

import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.phrasetable.MosesPhraseTable.TargetSideTranslation;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MosesPhraseTableTest
{

	@Test
	public void testPhraseTable() {
		final MosesPhraseTable readFromFile = MosesPhraseTable.readFromFile(FileUtils.getFile("test_phrase_table.moses").getPath());
		{
			final int[] array1 = WordIndexer.StaticMethods.toArrayFromStrings(readFromFile.getWordIndexer(), Arrays.asList("i", "like"));
			final List<TargetSideTranslation> translations = readFromFile.getTranslations(array1, 0, array1.length);
			assertEquals(3, translations.size());
			assertEquals(1, translations.get(2).trgWords.length);
			assertEquals(2, translations.get(0).trgWords.length);
		}

		{
			final int[] array1 = WordIndexer.StaticMethods.toArrayFromStrings(readFromFile.getWordIndexer(), List.of("i"));
			final List<TargetSideTranslation> translations = readFromFile.getTranslations(array1, 0, array1.length);
			assertEquals(1, translations.size());
			assertEquals(1, translations.get(0).trgWords.length);
		}

		{
			final int[] array1 = WordIndexer.StaticMethods.toArrayFromStrings(readFromFile.getWordIndexer(), List.of("want"));
			final List<TargetSideTranslation> translations = readFromFile.getTranslations(array1, 0, array1.length);
			assertEquals(0, translations.size());
		}
	}

	public static void main(final String[] argv) {
		new MosesPhraseTableTest().testPhraseTable();
	}
}
