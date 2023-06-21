package edu.berkeley.nlp.lm.io;

import edu.berkeley.nlp.lm.StupidBackoffLm;
import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.map.NgramsForOrderMapWrapper;
import edu.berkeley.nlp.lm.util.LongRef;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import static org.junit.jupiter.api.Assertions.*;

public class JavaMapWrapperTest
{

	public static void main(String[] argv) {
		new JavaMapWrapperTest().testBothMapWrapper();
	}

	@Test
	public void testBothMapWrapper() {
		final StupidBackoffLm<String> lm = (StupidBackoffLm<String>) LmReaders.readLmFromGoogleNgramDir(FileUtils.getFile("googledir").getPath(), false, false);
		final StupidBackoffLm<String> lm2 = (StupidBackoffLm<String>) LmReaders.readLmFromGoogleNgramDir(FileUtils.getFile("googledir").getPath(), true, false);
		final Map<List<String>, LongRef> m = LmReaders.readNgramMapFromGoogleNgramDir(FileUtils.getFile("googledir").getPath(), true);

		final List<String> ngram1 = List.of(",", "the", "(");
		final int[] array1 = WordIndexer.StaticMethods.toArray(lm.getWordIndexer(), ngram1);
		final List<String> ngram2 = List.of("the", "the", "(");
		final int[] array2 = WordIndexer.StaticMethods.toArray(lm.getWordIndexer(), ngram2);
		final List<String> ngram3 = List.of("the");
		final int[] array3 = WordIndexer.StaticMethods.toArray(lm.getWordIndexer(), ngram3);
		assertEquals(m.get(ngram1).value, 50);
		assertTrue(m.containsKey(ngram1));
		assertFalse(m.containsKey(ngram2));
		assertEquals(m.get(ngram3).value, 19401194714L);
		assertEquals(m.get(ngram1).value, 50);
		assertEquals(m.get(ngram3).value, 19401194714L);
		long totalSize = 0;
		for (int order = 0; order < lm.getLmOrder(); ++order) {
			final NgramsForOrderMapWrapper<String, LongRef> map = new NgramsForOrderMapWrapper<>(lm.getNgramMap(), lm.getWordIndexer(), order);
			final NgramsForOrderMapWrapper<String, LongRef> map2 = new NgramsForOrderMapWrapper<>(lm2.getNgramMap(), lm2.getWordIndexer(), order);
			if (order == 2) assertEquals(map.get(ngram1).value, 50);
			if (order == 2) assertEquals(lm.getRawCount(array1, 0, array1.length), 50);

			if (order == 2) assertEquals(lm.getRawCount(array2, 0, array2.length), -1L);
			if (order == 2) assertTrue(map.containsKey(ngram1));
			if (order == 2) assertFalse(map.containsKey(ngram2));
			if (order == 0) assertEquals(map.get(ngram3).value, 19401194714L);
			if (order == 0) assertEquals(lm.getRawCount(array3, 0, array3.length), 19401194714L);
			if (order == 2) assertEquals(map2.get(ngram1).value, 50);
			if (order == 0) assertEquals(map2.get(ngram3).value, 19401194714L);
			assertEquals(map.size(), map2.size());
			for (final Entry<List<String>, LongRef> entry : map.entrySet()) {

				final LongRef val = map2.get(entry.getKey());
				assertEquals(val, entry.getValue());
			}
			for (final Entry<List<String>, LongRef> entry : map2.entrySet()) {
				assertEquals(map.get(entry.getKey()), entry.getValue());
			}
			totalSize += map.size();

		}
		assertEquals(totalSize, m.size());

	}
}
