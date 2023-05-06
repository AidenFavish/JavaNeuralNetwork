import org.json.simple.JSONArray;

public class Matrix2D
{
    private float[][] matrix;

    public Matrix2D(float[][] grid) {
        matrix = grid;
    }

    public Matrix2D(int r, int c) {
        matrix = new float[r][c];
    }

    public static Matrix2D random(int r, int c) {
        float[][] ans = new float[r][c];
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                ans[i][j] = Random.getNext();
            }
        }
        return new Matrix2D(ans);
    }

    public Matrix2D multiplyConstant(float x) {
        float[][] ans = new float[matrix.length][matrix[0].length];

        for (int r = 0; r < matrix.length; r++) {
            for (int c = 0; c < matrix[0].length; c++) {
                ans[r][c] = matrix[r][c] * x;
            }
        }

        return new Matrix2D(ans);
    }

    public static Matrix2D zeros(int r, int c) {
        float[][] ans = new float[r][c];
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                ans[i][j] = 0;
            }
        }
        return new Matrix2D(ans);
    }

    public Matrix2D dot(Matrix2D x) {
        float[][] ans = new float[matrix.length][x.getMatrix()[0].length];
        float temp;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < x.getMatrix()[0].length; j++) {
                temp = 0;
                for (int y = 0; y < matrix[i].length; y++) {
                    temp += matrix[i][y] * x.getMatrix()[y][j];
                }
                ans[i][j] = temp;
            }
        }

        return new Matrix2D(ans);
    }

    public Matrix2D plus(Matrix2D x) {
        float[][] ans = new float[1][1];

        if (x.getMatrix().length == matrix.length && x.getMatrix()[0].length == matrix[0].length) {
            ans = new float[matrix.length][matrix[0].length];
            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix[0].length; j++) {
                    ans[i][j] = matrix[i][j] + x.getMatrix()[i][j];
                }
            }
        } else if (x.getMatrix().length == 1 && x.getMatrix()[0].length == matrix[0].length) {
            ans = new float[matrix.length][matrix[0].length];
            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix[0].length; j++) {
                    ans[i][j] = matrix[i][j] + x.getMatrix()[0][j];
                }
            }
        } else if (x.getMatrix().length == matrix.length && x.getMatrix()[0].length == 1) {
            ans = new float[matrix.length][matrix[0].length];
            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix[0].length; j++) {
                    ans[i][j] = matrix[i][j] + x.getMatrix()[i][0];
                }
            }
        } else if (matrix.length == 1 && x.getMatrix()[0].length == matrix[0].length) {
            ans = new float[x.getMatrix().length][x.getMatrix()[0].length];
            for (int i = 0; i < x.getMatrix().length; i++) {
                for (int j = 0; j < x.getMatrix()[0].length; j++) {
                    ans[i][j] = matrix[0][j] + x.getMatrix()[i][j];
                }
            }
        } else if (x.getMatrix().length == matrix.length && 1 == matrix[0].length) {
            ans = new float[x.getMatrix().length][x.getMatrix()[0].length];
            for (int i = 0; i < x.getMatrix().length; i++) {
                for (int j = 0; j < x.getMatrix()[0].length; j++) {
                    ans[i][j] = matrix[i][0] + x.getMatrix()[i][j];
                }
            }
        } else {
            System.out.println("ahhhh plus error!");
        }

        return new Matrix2D(ans);
    }

    public Matrix2D plus(float x) {
        float[][] ans = new float[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                ans[i][j] = matrix[i][j] + x;
            }
        }

        return new Matrix2D(ans);
    }

    public float[][] getMatrix() {
        return matrix;
    }

    @Override
    public String toString() {
        String str = "[";
        for (int i = 0; i < matrix.length; i++) {
            str += "[";
            for (int j = 0; j < matrix[0].length; j++) {
                str += " " + matrix[i][j] + " ";
            }
            str += "]" + (i < matrix.length-1 ? "\n" : "");
        }
        return str + "]";
    }

    public Matrix2D T() {
        float[][] ans = new float[matrix[0].length][matrix.length];

        for (int i = 0; i < ans.length; i++) {
            for (int j = 0; j < ans[0].length; j++) {
                ans[i][j] = matrix[j][i];
            }
        }

        return new Matrix2D(ans);
    }

    public Matrix2D sum(int axis) {
        if (axis == 0) {
            float[][] ans = new float[1][matrix[0].length];

            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix[0].length; j++) {
                    ans[0][j] += matrix[i][j];
                }
            }

            return new Matrix2D(ans);
        } else {
            float[][] ans = new float[matrix.length][1];

            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix[0].length; j++) {
                    ans[i][0] += matrix[i][j];
                }
            }

            return new Matrix2D(ans);
        }
    }

    public static Matrix2D onesLike(Matrix2D x) {
        Matrix2D ans = new Matrix2D(x.getMatrix().length, x.getMatrix()[0].length);

        for (int i = 0; i < ans.getMatrix().length; i++) {
            for (int j = 0; j < ans.getMatrix()[0].length; j++) {
                ans.getMatrix()[i][j] = 1;
            }
        }

        return ans;
    }

    public static Matrix2D zerosLike(Matrix2D x) {
        Matrix2D ans = new Matrix2D(x.getMatrix().length, x.getMatrix()[0].length);

        for (int i = 0; i < ans.getMatrix().length; i++) {
            for (int j = 0; j < ans.getMatrix()[0].length; j++) {
                ans.getMatrix()[i][j] = 0;
            }
        }

        return ans;
    }

    public Matrix2D conditionModify1(Matrix2D x, float y, float z) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (x.getMatrix()[i][j] <= y) {
                    matrix[i][j] = z;
                }
            }
        }

        return this;
    }

    public Matrix2D conditionModify2(Matrix2D x, float y, float z) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (x.getMatrix()[i][j] > y) { // >= was deliberately chosen not to be used
                    matrix[i][j] = z;
                }
            }
        }

        return this;
    }

    public Matrix2D copy() {
        float[][] ans = new float[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                ans[i][j] = matrix[i][j];
            }
        }
        return new Matrix2D(ans);
    }

    public float mean() {
        float sum = 0;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                sum += matrix[i][j];
            }
        }
        return sum / (matrix.length * matrix[0].length);
    }

    public Matrix2D exp() {
        float[][] ans = new float[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                ans[i][j] = (float)(Math.pow(Math.E, matrix[i][j]));
            }
        }

        return new Matrix2D(ans);
    }

    public Matrix2D max(int axis) {
        float[][] ans = new float[axis == 0 ? 1 : matrix.length][axis == 0 ? matrix[0].length : 1];
        float max;
        if (axis == 0) {
            for (int c = 0; c < matrix[0].length; c++) {
                max = matrix[0][c];
                for (int r = 1; r < matrix.length; r++) {
                    if (matrix[r][c] > max)
                        max = matrix[r][c];
                }
                ans[0][c] = max;
            }
        } else {
            for (int r = 0; r < matrix.length; r++) {
                max = matrix[r][0];
                for (int c = 1; c < matrix[0].length; c++) {
                    if (matrix[r][c] > max)
                        max = matrix[r][c];
                }
                ans[r][0] = max;
            }
        }
        return new Matrix2D(ans);
    }

    public Matrix2D divideBy(Matrix2D x) {
        float[][] ans = new float[1][1];

        if (x.getMatrix().length == matrix.length && x.getMatrix()[0].length == matrix[0].length) {
            ans = new float[matrix.length][matrix[0].length];
            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix[0].length; j++) {
                    ans[i][j] = matrix[i][j] / x.getMatrix()[i][j];
                }
            }
        } else if (x.getMatrix().length == 1 && x.getMatrix()[0].length == matrix[0].length) {
            ans = new float[matrix.length][matrix[0].length];
            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix[0].length; j++) {
                    ans[i][j] = matrix[i][j] / x.getMatrix()[0][j];
                }
            }
        } else if (x.getMatrix().length == matrix.length && x.getMatrix()[0].length == 1) {
            ans = new float[matrix.length][matrix[0].length];
            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix[0].length; j++) {
                    ans[i][j] = matrix[i][j] / x.getMatrix()[i][0];
                }
            }
        }

        return new Matrix2D(ans);
    }

    public Matrix2D pow(float x) {
        float[][] ans = new float[matrix.length][matrix[0].length];

        for (int r = 0; r < ans.length; r++) {
            for (int c = 0; c < ans[0].length; c++) {
                ans[r][c] = (float)(Math.pow(matrix[r][c], x));
            }
        }

        return new Matrix2D(ans);
    }

    public int[] argmax(int axis) {
        int[] ans = new int[axis == 1 ? matrix.length : matrix[0].length];
        int highestIndex;
        if (axis == 1) {
            for (int r = 0; r < matrix.length; r++) {
                highestIndex = 0;
                for (int c = 1; c < matrix[0].length; c++) {
                    if (matrix[r][highestIndex] < matrix[r][c])
                        highestIndex = c;
                }
                ans[r] = highestIndex;
            }
            return ans;
        }
        return null;
    }

    public Matrix2D abs() {
        float[][] ans = new float[matrix.length][matrix[0].length];
        for (int r = 0; r < matrix.length; r++) {
            for (int c = 0; c < matrix[0].length; c++) {
                ans[r][c] = matrix[r][c] < 0 ? -matrix[r][c] : matrix[r][c];
            }
        }
        return new Matrix2D(ans);
    }

    public float sum() {
        float ans = 0;
        for (int r = 0; r < matrix.length; r++) {
            for (int c = 0; c < matrix[0].length; c++) {
                ans += matrix[r][c];
            }
        }
        return ans;
    }

    public Matrix2D conditionOperate(float threshold, Operation foo) {
        for (int r = 0; r < matrix.length; r++) {
            for (int c = 0; c < matrix[0].length; c++) {
                matrix[r][c] = matrix[r][c] <= threshold ? foo.operate(matrix[r][c]) : matrix[r][c];
            }
        }

        return this;
    }

    public Matrix2D times(Matrix2D other) {
        float[][] ans = new float[matrix.length][matrix[0].length];
        for (int r = 0; r < matrix.length; r++) {
            for (int c = 0; c < matrix[0].length; c++) {
                ans[r][c] = matrix[r][c] * other.getMatrix()[r][c];
            }
        }

        return new Matrix2D(ans);
    }

    public JSONArray getJSON() {
        JSONArray outer = new JSONArray();
        JSONArray inner;

        for (int r = 0; r < matrix.length; r++) {
            inner = new JSONArray();
            for (int c = 0; c < matrix[0].length; c++) {
                inner.add(matrix[r][c]);
            }
            outer.add(inner);
        }

        return outer;
    }

}
