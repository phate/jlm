
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
