//=============================================================================================
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Conforti Christian
// Neptun : F8R430
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
// Forrasmegjeloles:
// A proceduralis texturazashoz az alabbi weboldalon talalhato informaciokat hasznaltam fel:
// http://www.upvector.com/?section=Tutorials&subsection=Intro%20to%20Procedural%20Textures&fbclid=IwAR0ZCaVUZh5mMhdhRLjruAKESTfuczaR-qGTDDGOlVQQvr6MJOb1CDwMJ70
//=============================================================================================

#include "framework.h"

#pragma region Constants

const int TESSELATION_LEVEL = 30;
const float PI_F = 3.14159265358979f;
const mat4 ENTITY_MATRIX = mat4{ { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };

// ASCII codes for used characters
const int x_KEY = 120;
const int y_KEY = 121;
const int z_KEY = 122;

const int X_KEY = 88;
const int Y_KEY = 89;
const int Z_KEY = 90;

#pragma endregion

#pragma region Input

struct Input {
	bool glutKeyTable[256];

	Input() {
		for (int i = 0; i < 256; i++) glutKeyTable[i] = false;
	}

	bool GetKeyStatus(int keyCode) { return glutKeyTable[keyCode]; }
};

Input input;

#pragma endregion

#pragma region Utility



#pragma endregion

#pragma region DualNumber

template<class T>
struct DualNumber {
	float funcVal;
	T derivative;

	DualNumber(float f0 = 0, T d0 = T(0)) {
		funcVal = f0;
		derivative = d0;
	}

	DualNumber operator+(const DualNumber& rhs) const {
		return DualNumber(funcVal + rhs.funcVal, derivative + rhs.derivative);
	}

	DualNumber operator-(const DualNumber& rhs) const {
		return DualNumber(funcVal - rhs.funcVal, derivative - rhs.derivative);
	}

	DualNumber operator*(const DualNumber& rhs) const {
		return DualNumber(funcVal * rhs.funcVal, funcVal * rhs.derivative + derivative * rhs.funcVal);
	}

	DualNumber operator/(const DualNumber& rhs) const {
		return DualNumber(funcVal / rhs.funcVal, (rhs.funcVal * derivative - rhs.derivative * funcVal) / (rhs.funcVal * rhs.funcVal));
	}
};

template<class T> DualNumber<T> Exp(DualNumber<T> g) { return DualNumber<T>(expf(g.funcVal), expf(g.funcVal) * g.derivative); }
template<class T> DualNumber<T> Sin(DualNumber<T> g) { return DualNumber<T>(sinf(g.funcVal), cosf(g.funcVal) * g.derivative); }
template<class T> DualNumber<T> Cos(DualNumber<T> g) { return DualNumber<T>(cosf(g.funcVal), -sinf(g.funcVal) * g.derivative); }
template<class T> DualNumber<T> Sinh(DualNumber<T> g) { return DualNumber<T>(sinh(g.funcVal), cosh(g.funcVal) * g.derivative); }
template<class T> DualNumber<T> Cosh(DualNumber<T> g) { return DualNumber<T>(cosh(g.funcVal), sinh(g.funcVal) * g.derivative); }
template<class T> DualNumber<T> Tanh(DualNumber<T> g) { return Sinh(g) / Cosh(g); }

using DualNumber2 = DualNumber<vec2>;

#pragma endregion


#pragma region Material and Light

struct Material {
	vec3 kd, ks, ka;
	float shininess;

	Material() { }
	Material(const vec3& _kd, const vec3& _ks, const vec3& _ka, const float& _shininess = 30) : kd{ _kd }, ka{ _ka }, ks{ _ks }, shininess{ _shininess } { }
};

struct Light {
	vec3 La, Le;
	vec4 wLightPos;
};

#pragma endregion

#pragma region Texture

class CheckerBoardTexture : public Texture {
public:
	CheckerBoardTexture(const int width, const int height, const vec4& color1 = vec4{ 1, 1, 0, 1 }, const vec4& color2 = vec4{ 0, 0, 1, 1 }) : Texture() {
		std::vector<vec4> image(width * height);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? color1 : color2;
		}
		create(width, height, image, GL_NEAREST);
	}
};

class StripeTexture : public Texture {
public:
	StripeTexture(const int width, const int height, const vec4& color1 = vec4{ 1, 1, 0, 1 }, const vec4& color2 = vec4{ 0, 0, 1, 1 }) : Texture() {
		std::vector<vec4> image(width * height);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = ((1 + sin(x * 50.0)) / 2) * color1;
		}
		create(width, height, image, GL_NEAREST);
	}
};

class FadedTexture : public Texture {
public:
	FadedTexture(const int width, const int height, const vec4& color = vec4{ 1, 1, 0, 1 }) : Texture() {
		std::vector<vec4> image(width * height);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = y * color;
		}
		create(width, height, image, GL_NEAREST);
	}
};

#pragma endregion

#pragma region Shader

struct RenderState {
	mat4 MVP, M, Minv, V, P;
	Material* material;
	std::vector<Light> lights;
	Texture* texture;
	vec3 wEye;
};

class PhongShader : public GPUProgram {
private:
	const char* vertexSource = R"(
		#version 330
		precision highp float;
		
		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4 MVP, M, Minv;
		uniform Light[8] lights;
		uniform int nLights;
		uniform vec3 wEye;

		layout(location = 0) in vec3 vtxPos;
		layout(location = 1) in vec3 vtxNorm;
		layout(location = 2) in vec2 vtxUV;

		out vec3 wNormal;
		out vec3 wView;
		out vec3 wLight[8];
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP;
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
			wView = wEye * wPos.w - wPos.xyz;
			wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
			texcoord = vtxUV;
		}
	)";

	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;
		uniform int nLights;
		uniform sampler2D diffuseTexture;

		in vec3 wNormal;
		in vec3 wView;
		in vec3 wLight[8];
		in vec2 texcoord;

		out vec4 fragmentColor;

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView);
			if (dot(N, V) < 0) N = -N;
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;
			
			vec3 radiance = vec3(0, 0, 0);	
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N, L), 0), cosd = max(dot(N, H), 0);
				radiance += ka * lights[i].La +
							(kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";

public:
	PhongShader() {
		create(vertexSource, fragmentSource, "fragmentColor");
	}

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}

	void bind(const RenderState& state) {
		Use();
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

PhongShader* gpuProgram;

#pragma endregion

#pragma region Camera

struct Camera {
	vec3 wEye, wLookat, wVup;
	float fov, asp, fp, bp;

public:

	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 20;
	}

	void set(const vec3& _wEye, const vec3& _wLookat, const vec3& _wVup) {
		wEye = _wEye;
		wLookat = _wLookat;
		wVup = _wVup;
	}

	mat4 V() {
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(vec3{ -wEye.x, -wEye.y, -wEye.z }) * mat4 {
			{ u.x, v.x, w.x, 0 },
			{ u.y, v.y, w.y, 0 },
			{ u.z, v.z, w.z, 0 },
			{ 0, 0, 0, 1 }
		};
	}

	mat4 P() {
		float sy = 1 / tanf(fov / 2);
		return mat4{
			{sy / asp, 0, 0, 0},
			{0, sy, 0, 0},
			{0, 0, -(fp + bp) / (bp - fp), -1},
			{0, 0, -2 * fp * bp / (bp - fp), 0}
		};
	}
};

#pragma endregion

#pragma region Geometry

struct VertexData {
	vec3 position, normal;
	vec2 textureCoordinate;

	VertexData() { }
	VertexData(vec3 _pos, vec3 _norm, vec2 _tex) : position{ _pos }, normal{ _norm }, textureCoordinate{ _tex } { }
};

class Geometry {
protected:
	unsigned int vao;	// Vertex Array Object 
	unsigned int vbo;	// Vertex Buffer Object 

public:
	Geometry() {
		glGenVertexArrays(1, &vao); // Can't be global variable
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}

	virtual void create(const float t = 0, const int N = TESSELATION_LEVEL, const int M = TESSELATION_LEVEL) = 0;
	virtual void draw() = 0;

	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

class RecursiveTetrahedron : public Geometry {
private:
	std::vector<VertexData> vertices;
	float length;
	float height;

	vec3 getNormalVector(VertexData& v1, VertexData& v2, VertexData& v3) const {
		return cross(normalize(v1.position - v2.position), normalize(v1.position - v3.position));
	}

	vec3 getEdgeMidpoint(const VertexData& v1, const VertexData& v2) const {
		return vec3{ (v1.position.x + v2.position.x) / 2, (v1.position.y + v2.position.y) / 2 , (v1.position.z + v2.position.z) / 2 };
	}

	vec3 getTriangleCentroid(const VertexData& v1, const VertexData& v2, const VertexData& v3) const {
		return vec3{ (v1.position.x + v2.position.x + v3.position.x) / 3, (v1.position.y + v2.position.y + v3.position.y) / 3, (v1.position.z + v2.position.z + v3.position.z) / 3 };
	}

	void getNewTetrahedron(VertexData& vertex1, VertexData& vertex2, VertexData& vertex3, int depth = 0) {

		if (depth > 1) return;

		VertexData nt1 = { getEdgeMidpoint(vertex1, vertex2), vec3{ 0, 0, 1 }, vec2{ 1, 1 } };
		VertexData nt2 = { getEdgeMidpoint(vertex2, vertex3), vec3{ 0, 0, 1 }, vec2{ 1, 1 } };
		VertexData nt3 = { getEdgeMidpoint(vertex3, vertex1), vec3{ 0, 0, 1 }, vec2{ 1, 1 } };
		VertexData nt4 = { getTriangleCentroid(vertex1, vertex2, vertex3) + normalize(getNormalVector(vertex1, vertex2, vertex3)) * height * 1 / (depth + 3), vec3{ 0, 0, 1 }, vec2{ 1, 1 } };

		nt1.normal = nt2.normal = nt4.normal = getNormalVector(nt1, nt2, nt4);

		vertices.push_back(nt1);
		vertices.push_back(nt2);
		vertices.push_back(nt4);

		nt2.normal = nt3.normal = nt4.normal = getNormalVector(nt2, nt3, nt4);

		vertices.push_back(nt2);
		vertices.push_back(nt3);
		vertices.push_back(nt4);

		nt1.normal = nt3.normal = nt4.normal = getNormalVector(nt1, nt3, nt4);

		vertices.push_back(nt1);
		vertices.push_back(nt3);
		vertices.push_back(nt4);

		getNewTetrahedron(nt2, nt1, nt3, depth + 1);
		getNewTetrahedron(nt1, nt2, nt4, depth + 1);
		getNewTetrahedron(nt3, nt1, nt4, depth + 1);
		getNewTetrahedron(nt2, nt3, nt4, depth + 1);
	}

public:

	RecursiveTetrahedron(const float _length = 1.0f, const float _height = 1.0f) : length{ _length }, height{ _height } { create(); }

	void init() {
		float x = sqrtf(3) / 3 * length;
		float d = sqrtf(3) / 6 * length;

		VertexData vertex1{ vec3{ x, 0, 0 }, vec3{ 0, 0, 1 }, vec2{ 1, 1 } };
		VertexData vertex2{ vec3{ -d, length / 2, 0 }, vec3{ 0, 0, 1 }, vec2{ 1, 1 } };
		VertexData vertex3{ vec3{ -d, -length / 2, 0 }, vec3{ 0, 0, 1 }, vec2{ 1, 1 } };
		VertexData vertex4{ vec3{0, 0, 1.0f}, vec3{ 0, 0, 1 }, vec2{ 1, 1 } };

		vertex1.normal = vertex2.normal = vertex3.normal = getNormalVector(vertex2, vertex1, vertex3);
		vertices.push_back(vertex1); vertices.push_back(vertex2); vertices.push_back(vertex3);

		vertex1.normal = vertex2.normal = vertex4.normal = getNormalVector(vertex1, vertex2, vertex4);
		vertices.push_back(vertex1); vertices.push_back(vertex2); vertices.push_back(vertex4);

		vertex2.normal = vertex3.normal = vertex4.normal = getNormalVector(vertex2, vertex3, vertex4);
		vertices.push_back(vertex2); vertices.push_back(vertex3); vertices.push_back(vertex4);

		vertex1.normal = vertex3.normal = vertex4.normal = getNormalVector(vertex3, vertex1, vertex4);
		vertices.push_back(vertex1); vertices.push_back(vertex3); vertices.push_back(vertex4);

		getNewTetrahedron(vertex2, vertex1, vertex3);
		getNewTetrahedron(vertex1, vertex2, vertex4);
		getNewTetrahedron(vertex3, vertex1, vertex4);
		getNewTetrahedron(vertex2, vertex3, vertex4);
	}

	void setHeight(float _height) { height = _height; }

	void create(const float t = 0, const int N = TESSELATION_LEVEL, const int M = TESSELATION_LEVEL) override {

		init();

		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(VertexData), &vertices[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), reinterpret_cast<void*>(offsetof(VertexData, position)));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), reinterpret_cast<void*>(offsetof(VertexData, normal)));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), reinterpret_cast<void*>(offsetof(VertexData, textureCoordinate)));
	}

	void draw() override {
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, vertices.size());
	}
};

class ParametricSurface : public Geometry {
private:
	unsigned int vertexPerStrip, nStrips;

public:
	ParametricSurface() : vertexPerStrip{ 0 }, nStrips{ 0 } { }

	virtual void evaluate(DualNumber2& U, DualNumber2& V, DualNumber2& X, DualNumber2& Y, DualNumber2& Z, const float t) = 0;

	VertexData genVertexData(const float u, const float v, const float t) {
		VertexData vertexData;
		vertexData.textureCoordinate = vec2(u, v);

		DualNumber2 X, Y, Z;
		DualNumber2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		evaluate(U, V, X, Y, Z, t);

		vertexData.position = vec3(X.funcVal, Y.funcVal, Z.funcVal);
		vertexData.normal = cross(vec3{ X.derivative.x, Y.derivative.x, Z.derivative.x }, vec3{ X.derivative.y, Y.derivative.y, Z.derivative.y });

		return vertexData;
	}
	
	void create(const float t = 0, const int N = TESSELATION_LEVEL, const int M = TESSELATION_LEVEL) override {
		vertexPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vertices;

		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vertices.push_back(genVertexData((float)j / M, (float)i / N, t));
				vertices.push_back(genVertexData((float)j / M, (float)(i + 1) / N, t));
			}
		}

		glBufferData(GL_ARRAY_BUFFER, vertexPerStrip * nStrips * sizeof(VertexData), &vertices[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);	// Attribute Array 0 = Position
		glEnableVertexAttribArray(1);	// Attribute Array 1 = Normal Vector
		glEnableVertexAttribArray(2);	// Attribute Array 2 = Texture Coordinate

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), reinterpret_cast<void*>(offsetof(VertexData, position)));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), reinterpret_cast<void*>(offsetof(VertexData, normal)));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), reinterpret_cast<void*>(offsetof(VertexData, textureCoordinate)));
	}

	void draw() override {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) {
			glDrawArrays(GL_TRIANGLE_STRIP, i * vertexPerStrip, vertexPerStrip);
		}
	}
};

class Sphere  : public ParametricSurface {
private:
	virtual DualNumber2 R(DualNumber2& U, DualNumber2& V, const float t) {
		return 1.0f;
	}

public:
	Sphere() { create(); }

	void evaluate(DualNumber2& U, DualNumber2& V, DualNumber2& X, DualNumber2& Y, DualNumber2& Z, const float t) override final {
		U = U * 2.0f * PI_F;
		V = V * PI_F;
		X = Cos(U) * Sin(V) * 2.0f * R(U, V, t);
		Y = Sin(U) * Sin(V) * 2.0f * R(U, V, t);
		Z = Cos(V) * 2.0f * R(U, V, t);
	}
};

class AngrySphere final : public Sphere {
private:
	DualNumber2 R(DualNumber2& U, DualNumber2& V, const float t) override final {
		float time = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
		return Sin(Cos(Sin(Cos(U + DualNumber2(3.0f) * V + 3.0f * time))));
		//return 1.0f + t;
	}

public:
	AngrySphere() { create(); }
};

class Tractricoid final : public ParametricSurface {
public:
	Tractricoid() { create(); }

	void evaluate(DualNumber2& U, DualNumber2& V, DualNumber2& X, DualNumber2& Y, DualNumber2& Z, const float t) override final {
		const float height = 4.0f;
		U = U * height;
		V = V * 2.0f * PI_F;
		X = Cos(V) / Cosh(U);
		Y = Sin(V) / Cosh(U);
		Z = U - Tanh(U);
	}
};

#pragma endregion

#pragma region Object

class GameObject {
protected:
	Material* material;
	Texture* texture;
	Geometry* geometry;

	// Model transformation parameters
	vec3 scale, translation, rotationAxis;
	float rotationAngle;

public:

	GameObject(Material* _material, Texture* _texture, Geometry* _geometry) : scale{ vec3{ 1, 1, 1 } }, translation{ vec3{ 0, 0, 0 } }, rotationAxis{ vec3{ 0, 0, 1 } }, rotationAngle{ 0 } {
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}

	void setScale(const vec3& _scale) { scale = _scale; }
	void setTranslation(const vec3& _translation) { translation = _translation; }
	void setRotationAxis(const vec3& _axis) { rotationAxis = _axis; }
	void setRotationAngle(const float& _angle) { rotationAngle = _angle; }

	vec3 getTranslation() { return translation; }
	Geometry* getGeometry() { return geometry; }
	Material* getMaterial() { return material; }
	Texture* getTexture() { return texture; }

	virtual void setModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3{ 1 / scale.x, 1 / scale.y, 1 / scale.z });
	}

	virtual void draw(RenderState state) {
		mat4 M, Minv;
		setModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		gpuProgram->bind(state);
		geometry->draw();
	}

	virtual void animate(const float t) { }
};

class Virus final : public GameObject{
private:
	// Children elements
	std::vector<GameObject*> coronas;

	void createCorona(const float t = 0) {

		Tractricoid* trac = new Tractricoid();

		Material* material0 = new Material();
		material0->kd = vec3{ 0.9f, 0.4f, 0.2f };
		material0->ks = vec3{ 4, 4, 4 };
		material0->ka = vec3{ 0.3f, 0.3f, 0.3f };
		material0->shininess = 100;

		Texture* texture1 = new FadedTexture(4, 8, vec4{ 1, 0, 0, 1 });
		Sphere* s = new AngrySphere();

		int nStrips = 10;
		for (int i = 0; i <= nStrips; i++) {
			int coronaPerStrip = 20 * sinf(PI_F * i / nStrips);
			for (int j = 0; j <= coronaPerStrip; j++) {
				GameObject* corona = new GameObject(material0, texture1, trac);
				corona->setScale(vec3{ 0.18f, 0.18f, 0.18f });
				VertexData v = s->genVertexData((float)j / coronaPerStrip, (float)i / nStrips, t);
				vec3 n = normalize(v.normal);
				corona->setRotationAxis(cross(n, vec3{0, 0, -1}));
				corona->setRotationAngle(acosf(dot(vec3{0, 0, 1}, n)));
				corona->setTranslation(1.2f * v.position + this->getTranslation());
				coronas.push_back(corona);
			}
		}
	}

public:
	Virus(Material* _material, Texture* _texture, Geometry* _geometry) : GameObject{ _material, _texture, _geometry } {
		createCorona();
	}

	void setModelingTransform(mat4& M, mat4& Minv) override final {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3{ 1 / scale.x, 1 / scale.y, 1 / scale.z });
	}

	void draw(RenderState state) override final {
		mat4 M, Minv;
		setModelingTransform(M, Minv);
		GameObject::draw(state);

		mat4 Mt, Mtinv;
		for (GameObject* c : coronas) {
			c->setModelingTransform(Mt, Mtinv);
			state.M = Mt * M;
			state.Minv = Minv * Mtinv;
			state.MVP = state.M * state.V * state.P;
			state.material = c->getMaterial();
			state.texture = c->getTexture();
			gpuProgram->bind(state);
			c->getGeometry()->draw();
		}
	}

	void animate(const float t) override {
		this->geometry->create(t);
		this->setRotationAngle(0.8 * t);
		//this->setScale(vec3{ sinf(t/4) * sinf(t/4) + 0.3f, sinf(t/4) * sinf(t/4) + 0.3f, sinf(t/4) * sinf(t/4) + 0.3f });
	}

};

class AntiBody final : public GameObject {
public:
	AntiBody(Material* _material, Texture* _texture, Geometry* _geometry) : GameObject{ _material, _texture, _geometry } { }

	void animate(const float t) override {

		delete geometry;
		geometry = new RecursiveTetrahedron(1.0f, sinf(t) + 1.0f);

		// Brown movement
		if (((int)(100 * t) % 10) == 0) {
			
			if (input.glutKeyTable[x_KEY]) {
				translation = translation + vec3(1.0f, 0.0f, 0.0f) * 0.05;
			}
			else if (input.glutKeyTable[X_KEY]) {
				translation = translation + vec3(-1.0f, 0.0f, 0.0f) * 0.05;
			}
			else {
				translation = translation + vec3(rand() % 3 - 1, rand() % 3 - 1, rand() % 3 - 1) * 0.05;
			}
		}
	}
};

#pragma endregion

#pragma region Scene

class Scene {
private:
	Camera camera;
	std::vector<GameObject*> objects;
	std::vector<Light> lights;

public:

	void build() {

		// Camera
		camera.set(vec3{ 0, 0, 8 }, vec3{ 0, 0, 0 }, vec3{ 0, 1, 0 });

		// Materials
		Material* material0 = new Material();
		material0->kd = vec3{ 0.6f, 0.4f, 0.2f };
		material0->ks = vec3{ 2, 2, 2 };
		material0->ka = vec3{ 0.1f, 0.1f, 0.1f };
		material0->shininess = 50;

		Material* material1 = new Material();
		material1->kd = vec3{ 0.8f, 0.6f, 0.4f };
		material1->ks = vec3{ 0.3f, 0.3f, 0.3f };
		material1->ka = vec3{ 0.2f, 0.2f, 0.2f };
		material1->shininess = 30;

		// Lights
		lights.resize(3);
		lights[0].wLightPos = vec4{ 5, 5, 4, 0 };
		lights[0].La = vec3{ 0.1f, 0.1f, 0.1f };
		lights[0].Le = vec3{ 1, 1, 1 };

		lights[1].wLightPos = vec4{ 5, 10, 20, 0 };
		lights[1].La = vec3{ 0.2f, 0.2f, 0.2f };
		lights[1].Le = vec3{ 1, 1, 1 };

		lights[2].wLightPos = vec4{ -5, 5, 5, 0 };
		lights[2].La = vec3{ 0.1f, 0.1f, 0.1f };
		lights[2].Le = vec3{ 1, 1, 1 };

		// Textures
		Texture* roomTexture = new CheckerBoardTexture(4, 8, vec4{ 1.0f, 0.0f, 1.0f, 1.0f }, vec4{ 1.0f, 0.0f, 1.0f, 1.0f });
		Texture* virusTexture = new StripeTexture(800, 1600);
		Texture* antiBodyTexture = new CheckerBoardTexture(4, 8, vec4{ 1.0f, 1.0f, 1.0f, 1.0f }, vec4{ 1.0f, 1.0f, 1.0f, 1.0f });

		// Geometries
		Geometry* sphere = new Sphere();
		Geometry* angrySphere = new AngrySphere();
		Geometry* tetrahedron = new RecursiveTetrahedron();

		// Objects

		// Room
		GameObject* room = new GameObject(material1, roomTexture, sphere);
		room->setScale(vec3{ 5.0f, 5.0f, 5.0f });
		objects.push_back(room);

		// Virus
		GameObject* virus = new Virus(material0, virusTexture, angrySphere);
		virus->setRotationAxis(vec3{ 1.0f, 0.0f, 0.0f });	
		virus->setTranslation(vec3{ -2.0f, -2.0f, 0.0f });
		objects.push_back(virus);

		// Antibody
		GameObject* antiBody = new AntiBody(material0, antiBodyTexture, tetrahedron);
		antiBody->setRotationAxis(vec3{ 1.0f, 0.0f, 0.0f });
		antiBody->setScale(vec3{ 1.5f, 1.5f, 1.5f });
		antiBody->setTranslation(vec3{ 1.0f, 1.0f, 0.0f });
		objects.push_back(antiBody);

	}

	void render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.M = ENTITY_MATRIX;
		state.Minv = ENTITY_MATRIX;
		state.lights = lights;

		for (auto object : objects) {
			object->draw(state);
		}
	}

	void animate(const float t) {
		for (auto object : objects) {
			object->animate(t);
		}
	}
};

Scene scene;

#pragma endregion

# pragma region EventHandling

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	gpuProgram = new PhongShader();
	scene.build();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);				// clear frame buffer

	scene.render();

	glutSwapBuffers(); // exchange buffers for double buffering
}

void onKeyboard(unsigned char key, int pX, int pY) { 
	input.glutKeyTable[key] = true;
}

void onKeyboardUp(unsigned char key, int pX, int pY) { 
	input.glutKeyTable[key] = false;
}

void onMouseMotion(int pX, int pY) { }

void onMouse(int button, int state, int pX, int pY) { }

void onIdle() {
	// Discrete time simulation
	static float tend = 0;
	const float dt = 0.1f;
	float tstart = tend;

	// Conversion to seconds
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	
	for (float t = tstart; t < tend; t += dt) {
		float Dt = fminf(dt, tend - t);
		scene.animate(t);
	}

	// Redrawing the display
	glutPostRedisplay();
}

#pragma endregion
	