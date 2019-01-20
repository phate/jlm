int
f(int x)
{
	if (x == 75)
		goto label;

	do {
		x++;
		label:
		if (x == 2)
			return x;

		if (x == 4)
			return x;

	} while (x--);

	return 0;
}

int
irreducible(int x)
{
	if (x < 50) {
		if (x == 2)
			goto label;
	} else {
		do {
			if (x) {
				label:
				x++;
			}
		} while (1);
	}

	return x;
}

int
main()
{
	return 0;
}
