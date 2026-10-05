"""Small CPU-only checks for the conservative quantity guard."""
from quantity_guard import check


def test_supported_equivalents_and_corruption():
    for ja, en in (
        ('5キロメートル', '5000 meters'),
        ('3平方キロメートル', '3000000 m²'),
        ('3平方キロメートル', '3 square kilometers'),
        ('三キログラム', '3000 grams'),
        ('三キログラム', '3 kilograms'),
        ('五分', '5 minutes'),
        ('0.5%', '0.5 percent'),
        ('3月5日', 'March 5'),
        ('午後3時', '3 PM'),
        ('午後五時四十分', '5:40 PM'),
        ('1,000円', '1000 yen'),
    ):
        assert check(ja, en)['status'] == 'consistent', (ja, en, check(ja, en))
    assert check('5キロメートル', '6 kilometers')['status'] == 'mismatch'
    assert check('1キログラム', '2 kilograms')['status'] == 'mismatch'


def test_unsupported_inputs_and_multiset_limit():
    for ja, en in (
        ('約5分', 'about 5 minutes'),
        ('5分から10分', '5 to 10 minutes'),
        ('五百円から六百円', '500 to 600 yen'),
        ('五百円〜六百円', '500 through 600 yen'),
        ('−五円', '-5 yen'),
        ('-五円', '-5 yen'),
        ('一万万キログラム', '10000 kilograms'),
        ('5キログラム', '5 grandchildren'),
        ('5時間', 'five hourglasses'),
        ('零百グラム', '100 grams'),
        ('3月1日', 'March 1.5'),
        ('午後1時', '1.5 PM'),
        ('一時間三十分', '90 minutes'),
        ('2メートル30センチメートル', '1 meter and 130 centimeters'),
        ('1キログラム200グラム', '1200 grams'),
        ('Aさんは5円、Bさんは5円払う。', 'A and B each pay 5 yen.'),
        ('5キロ', '5000 meters'),
        ('3センチリットル', '30 milliliters'),
        ('時速5キロメートル', '5 km/h'),
        ('5 cm/s', '5 centimeters per second'),
        ('1キログラム', 'one and two kilograms'),
        ('午後3時', 'five fifteen PM'),
        ('2月30日', 'February 30'),
        ('午後13時', '1 PM'),
        ('二百百キログラム', '300 kilograms'),
        ('半分', 'half'),
        ('彼は二人です', 'There are two people.'),
        ('今日は休みです', 'Today is a day off.'),
        ('何もありません', 'There is nothing.'),
    ):
        assert check(ja, en)['status'] == 'abstain', (ja, en, check(ja, en))
    # A multiset cannot see who received which of two equal-valued quantities.
    result = check('Aさんに5円、Bさんに5円を渡した', 'I gave B 5 yen and A 5 yen.')
    assert result['status'] == 'consistent', result


if __name__ == '__main__':
    test_supported_equivalents_and_corruption()
    test_unsupported_inputs_and_multiset_limit()
    print('quantity_guard checks passed')
