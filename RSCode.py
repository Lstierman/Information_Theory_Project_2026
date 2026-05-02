import galois
import numpy as np

class RSCode:
    def __init__(self, m,t,l,m0):
        self.m = m #GF(2^m) field
        self.t = t #Error correction capability
        self.n = 2**m-1 #Code length
        self.k = self.n-2*t #Information length
        self.l = l #Shortened information length (-> shortened code length = l+n-k)
        self.m0 = m0 #m0 of the Reed-Solomon code, determines first root of generator
        
        self.g = self.makeGenerator(m,t,m0) # generator polynomial represented by a galois.Poly variable

    def encode(self,msg):
        # Systematically encodes information words using the Reed-Solomon code
        # Input:
        #  -msg: a 2D array of galois.GF elements, every row corresponds with a GF(2^m) information word of length self.l
        # Output:
        #  -code: a 2D array of galois.GF elements, every row contains a GF(2^m) codeword corresponding to systematic Reed-Solomon coding of the corresponding information word
        assert np.shape(msg)[1] == self.l, 'the number of columns must be equal to self.l'
        assert type(msg) is galois.GF(2**self.m) , 'each element of msg  must be a galois.GF element'

        #insert your code here
        GF = galois.GF(2**self.m)
        n_parity = self.n - self.k

        # Output: one shortened codeword per input row.
        code = GF.Zeros((np.shape(msg)[0], self.l + n_parity))

        for row_index in range(np.shape(msg)[0]):
            # Put the message first and reserve the last positions for parity.
            shifted_message = GF.Zeros(self.l + n_parity)
            shifted_message[:self.l] = msg[row_index, :]

            # Divide the shifted message polynomial by g(x).
            shifted_message_poly = galois.Poly(shifted_message, field=GF)
            remainder = shifted_message_poly % self.g

            # Systematic codeword: message followed by corrected parity part.
            code[row_index, :] = shifted_message
            if remainder.degree >= 0:
                code[row_index, -len(remainder.coeffs):] -= remainder.coeffs

        assert np.shape(code)[1] == self.l+self.n-self.k , 'the number of columns must be equal to self.l+self.n-self.k'
        assert type(code) is galois.GF(2**self.m) , 'each element of code  must be a galois.GF element'
        return code

    def decode(self,code):
        # Decode Reed-Solomon codes
        # Input:
        #  -code: a 2D array of galois.GF elements, every row contains a GF(2^m) codeword of length self.l+self.n-self.k
        # Output:
        #  -decoded: a 2D array of galois.GF elements, every row contains a GF(2^m) information word corresponding to decoding of the corresponding Reed-Solomon codeword
        #  -nERR: 1D numpy array containing the number of corrected symbols for every codeword, -1 if error correction failed
        assert np.shape(code)[1] == self.l+self.n-self.k , 'the number of columns must be equal to self.l+self.n-self.k'
        assert type(code) is galois.GF(2**self.m) , 'each element of code  must be a galois.GF element'

        #insert your code here
        # This decoder uses the following algebraic decoding techniques from the course notes:
        #  - Berlekamp-Massey algorithm, page 77:
        #    computes the error-locator polynomial Lambda(z).
        # (Berklekamp-Massey algorithm is used because it avoids trying all possible error-position combinations.)
        #  - Forney's algorithm, Section 2.4.6, page 82:
        #    computes the error values once the positions are known.
        GF = galois.GF(2**self.m)
        alpha = GF.primitive_element

        n_parity = self.n - self.k
        codeword_length = self.l + n_parity

        corrected = code.copy()
        nERR = np.zeros(np.shape(code)[0], dtype=int)

        # in the polynomial representation used by galois.Poly(word), position p corresponds to x^(codeword_length - 1 - p)
        pos_to_exp = np.array(
            [codeword_length - 1 - p for p in range(codeword_length)],
            dtype=int
        )

        # precompute the syndrome evaluation matrix
        # syndrome component i is r(alpha^(m0+i))
        eval_matrix = GF.Zeros((n_parity, codeword_length))
        for i in range(n_parity):
            root = alpha ** (self.m0 + i)
            for p, exp in enumerate(pos_to_exp):
                eval_matrix[i, p] = root ** exp

        def syndrome_of(word):
            # computes all 2t syndromes of one received word
            return eval_matrix @ word

        def trim(poly):
            # polynomial stored in ascending powers:
            # poly[0] + poly[1] z + ... + poly[d] z^d
            while len(poly) > 1 and poly[-1] == 0:
                poly = poly[:-1]
            return poly

        def poly_add(a, b):
            # adds two polynomials stored in ascending-power order
            length = max(len(a), len(b))
            out = GF.Zeros(length)
            out[:len(a)] += a
            out[:len(b)] += b
            return trim(out)

        def poly_mul(a, b):
            # multiplies two polynomials stored in ascending-power order
            out = GF.Zeros(len(a) + len(b) - 1)
            for i, ai in enumerate(a):
                if ai != 0:
                    out[i:i + len(b)] += ai * b
            return trim(out)

        def poly_eval_ascending(poly, x):
            # evaluates poly[0] + poly[1]x + ... + poly[d]x^d
            y = GF(0)
            power = GF(1)

            for coeff in poly:
                y += coeff * power
                power *= x

            return y

        def berlekamp_massey(S):
            """
            (Found in course notes page 77)
            Computes the error-locator polynomial Lambda(z).

            Lambda is stored in ascending powers:
                Lambda(z) = 1 + Lambda_1 z + ... + Lambda_L z^L
            """
            Lambda = GF([1])
            B = GF([0, 1])
            L = 0

            for i in range(1, n_parity + 1):
                # compute the discrepancy.
                delta = GF(0)

                for j in range(L + 1):
                    if j < len(Lambda):
                        delta += Lambda[j] * S[i - 1 - j]

                # non-zero discrepancy means Lambda(z) must be updated
                if delta != 0:
                    old_Lambda = Lambda.copy()

                    update = delta * B
                    Lambda = poly_add(Lambda, -update)

                    if 2 * L < i:
                        L = i - L
                        B = old_Lambda / delta

                # shift B(z) by one power of z
                B = np.concatenate((GF.Zeros(1), B))

                if L > self.t:
                    return None, -1

            return trim(Lambda), L

        def forney_error_value(S, Lambda, exponent):
            """
            (Found in course notes page 82)
            Computes the error value at exponent ell.

            If the error is at polynomial coefficient x^ell, then:
                X = alpha^ell
                Lambda(X^-1) = 0
            """
            X = alpha ** exponent
            z = X ** -1

            # Omega(z) = S(z) Lambda(z) mod z^(2t)
            S_poly = S.copy()
            Omega = poly_mul(S_poly, Lambda)

            if len(Omega) > n_parity:
                Omega = Omega[:n_parity]

            omega_z = poly_eval_ascending(Omega, z)

            # compute the formal derivative Lambda'(z)
            lambda_prime_z = GF(0)
            characteristic = GF.characteristic
            power = GF(1)

            for k in range(1, len(Lambda)):
                coeff = GF(k % characteristic) * Lambda[k]
                lambda_prime_z += coeff * power
                power *= z

            if lambda_prime_z == 0:
                raise ZeroDivisionError("Forney denominator is zero")

            # correction value for non-narrow-sense RS with first root m0
            return -omega_z * (X ** (1 - self.m0)) / lambda_prime_z

        for row_index in range(np.shape(code)[0]):
            received = code[row_index, :].copy()
            S = syndrome_of(received)

            # zero syndrome means no errors were detected
            if np.all(S == 0):
                corrected[row_index, :] = received
                nERR[row_index] = 0
                continue

            Lambda, L = berlekamp_massey(S)

            if Lambda is None or L < 0 or L > self.t:
                corrected[row_index, :] = received
                nERR[row_index] = -1
                continue

            # find the error positions from the roots of the error-locator polynomial      
            error_positions = []
            error_exponents = []

            for p, exp in enumerate(pos_to_exp):
                z = (alpha ** exp) ** -1

                if poly_eval_ascending(Lambda, z) == 0:
                    error_positions.append(p)
                    error_exponents.append(exp)

            # decoder failure if the locator degree does not match the number of roots
            if len(error_positions) != L:
                corrected[row_index, :] = received
                nERR[row_index] = -1
                continue

            candidate = received.copy()

            try:
                for p, exp in zip(error_positions, error_exponents):
                    error_value = forney_error_value(S, Lambda, exp)
                    candidate[p] -= error_value
            except ZeroDivisionError:
                corrected[row_index, :] = received
                nERR[row_index] = -1
                continue

            # final check: the corrected word must have zero syndrome
            if np.all(syndrome_of(candidate) == 0):
                corrected[row_index, :] = candidate
                nERR[row_index] = L
            else:
                corrected[row_index, :] = received
                nERR[row_index] = -1

        # the information part of the corrected codewords is the decoded message
        decoded = corrected[:, :self.l]

        assert np.shape(decoded)[1] == self.l, 'the number of columns must be equal to self.l'
        assert type(decoded) is galois.GF(2**self.m) , 'each element of decoded  must be a galois.GF element'
        assert type(nERR) is np.ndarray and len(np.shape(nERR))==1 , 'nERR must be a 1D numpy array'

        return (decoded,nERR)




    @staticmethod
    def makeGenerator(m, t, m0):
        # Generate the Reed-Solomon generator polynomial with error correcting capability t over GF(2^m)
        # Input:
        #  -m: order of the galois field is 2^m
        #  -t: error correction capability of the Reed-Solomon code
        #  -m0: determines the first root of the generator polynomial
        # Output:
        #  -generator: generator polynomial represented by a galois.Poly variable

        #insert your code here
        GF = galois.GF(2**m)
        alpha = GF.primitive_element

        # start with the constant polynomial 1
        generator = galois.Poly([1], field=GF)

        # construct the RS generator polynomial from its 2t consecutive roots:
        # alpha^m0, alpha^(m0+1), ..., alpha^(m0+2t-1).
        for i in range(2*t):
            generator *= galois.Poly([1, -(alpha ** (m0 + i))], field=GF)

        assert type(generator) == type(galois.Poly([0],field=galois.GF(2**m))), 'generator must be a galois.Poly object'
        return generator

    @staticmethod
    def test():
        # function that illustrates how the other code of this class can be tested
        m0 = 1 # Also test with other values of m0!
        m=8
        t=5
        l=10
        rs = RSCode(m,t,l,m0) # Construct the RSCode object
        p=2
        prim_poly=galois.primitive_poly(p,m)
        # I had a version of galois that required the primitive polynomial to be passed as a galois.Poly object, but the latest version accepts it as a list of coefficients. To be compatible with both versions, I try both ways here.
        try:
            galois_field = galois.GF(p**m, irreducible_poly=prim_poly)
        except TypeError:
            galois_field = galois.GF(p**m, prim_poly)


        msg = galois_field(np.random.randint(0,2**8-1,(5,10))) # Generate a random message of 5 information words

        code = rs.encode(msg) # Encode this message

        # Introduce errors
        code[1,[2, 17]] = code[1,[4, 17]]+galois_field(1)
        code[2,7] = 0;
        code[3,[3, 1, 18, 19, 5]] = np.random.randint(0,2**8-1,(1,5))
        code[4,[3, 1, 18, 19, 5, 12]] = np.random.randint(0,2**8-1,(1,6))


        [decoded,nERR] = rs.decode(code) # Decode


        print(nERR)
        assert((decoded[0:4,:] == msg[0:4,:]).all())
        pass