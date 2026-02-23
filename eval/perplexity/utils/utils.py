def fix_serialized_bytes(s: str) -> str:
    try:
        # strip leading b' and trailing '
        inner = s[2:-1]

        # turn escape sequences into raw bytes
        b = inner.encode("latin-1").decode("unicode_escape").encode("latin-1")

        # decode bytes properly
        return b.decode("utf-8")
    except:
        return s # the text is not a serialized byte string, return as is

if __name__ == "__main__":
    # test
    # text = "b'\n_START_ARTICLE_\nBarn Jams\n_START_PARAGRAPH_\n\xe2\x80\x9eBarn Jams\xe2\x80\x9c (vo vo\xc4\xbenom preklade: improviz\xc3\xa1cie v stodole) s\xc3\xba in\xc5\xa1trument\xc3\xa1lne pas\xc3\xa1\xc5\xbee (skladby), ktor\xc3\xa9 zlo\xc5\xbeil britsk\xc3\xbd hudobn\xc3\xadk a gitarista niekdaj\xc5\xa1ej rockovej skupiny Pink Floyd, David Gilmour. Existuj\xc3\xba aj z\xc3\xa1bery, na ktor\xc3\xbdch s\xc3\xba tieto skladby predstaven\xc3\xa9, hraj\xc3\xba na nich David Gilmour (gitary), Richard Wright (kl\xc3\xa1vesy), Guy Pratt (basgitara) a Steve DiStanislao (bicie). T\xc3\xadto hudobn\xc3\xadci s\xc3\xba taktie\xc5\xbe s\xc3\xba\xc4\x8das\xc5\xa5ou live kapely, s ktorou moment\xc3\xa1lne koncertuje David na Rattle That Lock Tour. Z\xc3\xa1bery boli nato\xc4\x8den\xc3\xa9 v janu\xc3\xa1ri 2007 v stodole, ktor\xc3\xba vlastn\xc3\xad gitarista._NEWLINE_Nie je presne zn\xc3\xa1me, ko\xc4\xbeko tak\xc3\xbdchto z\xc3\xa1znamov existuje, tri boli vydan\xc3\xa9 ako \xc4\x8das\xc5\xa5 4-diskovej, deluxe a vinylovej ed\xc3\xadcie albumu Live in Gda\xc5\x84sk (2008). \xc4\x8eal\xc5\xa1ia skladba bola s\xc3\xba\xc4\x8das\xc5\xa5ou albumu Remember That Night (2007) a \xc5\xa1tyri \xc4\x8fal\xc5\xa1ie boli s\xc3\xba\xc4\x8das\xc5\xa5ou deluxe ed\xc3\xadcie albumu Rattle That Lock (2015). Toto s\xc3\xba posledn\xc3\xa9 ofici\xc3\xa1lne z\xc3\xa1bery klaviristu, kl\xc3\xa1ves\xc3\xa1ka a \xc4\x8dlena skupiny Pink Floyd, Richarda Wrighta (\xe2\x80\xa0 2008).'"
    # print(fix_serialized_bytes(text))

    text = "_START_ARTICLE_\nŠubši-mašrâ-Šakkan\n_START_SECTION_\nThe sources\n_START_PARAGRAPH_\nA tablet recovered in Nippur lists grain rations given to the messenger of a certain Šubši-mašrâ-Šakkan during Nazi-Marrutaš’ fourth year (1304 BC). There is a court order found in Ur, dated to the sixteenth year of Nazi-Maruttaš (1292 BC), in which Šubši-mašrâ-šakkan is given the title šakin māti, lúGAR KUR, “governor of the country.” It is an injunction forbidding harvesting reeds from a certain river or canal._NEWLINE_The poetic work, Ludlul bēl nēmeqi, describes how the fortunes of Šubši-mašrâ-Šakkan, a rich man of high rank, turned one day. When beset by ominous signs, he incurred the wrath of the king, and seven courtiers plotted every kind of mischief against him. This resulted in him losing his property, “they have divided all my possessions among foreign riffraff,” friends, “my city frowns on me as an enemy; indeed my land is savage and hostile,” physical strength, “my flesh is flaccid, and my blood has ebbed away,” and health, as he relates that he “wallowed in my excrement like a sheep.” While slipping into and out of consciousness on his death bed, his family already conducting his funeral, Urnindinlugga, a kalû, or incantation priest, was sent by Marduk to presage his salvation. The work concludes with a prayer to Marduk. The text is written in the first person, leading some to speculate that the author was Šubši-mašrâ-Šakkan himself. Perhaps the only certainty is that the subject of the work, Šubši-mašrâ-Šakkan, was a significant historical person during the reign of Nazi-Maruttaš when the work was set. Of the fifty eight extant fragmentary copies of Ludlul bēl nēmeqi the great majority date to the neo-Assyrian and neo-Babylonian periods."
    print(fix_serialized_bytes(text))